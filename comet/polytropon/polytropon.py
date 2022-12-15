import logging
import math
from scipy import special
from comet.transformers_ptuning.ptuning_wrapper import * 

import torch
from torch import nn

from comet.polytropon.adapters import (
    HyperLoRALinear,
    SkilledLoRALinear,
    SkilledLTSFTLinear,
)
from comet.polytropon.utils import replace_layers, inform_layers

logger = logging.getLogger(__name__)

VARIANT2CLASS = {
    "hyperformer": (HyperLoRALinear, True),
    "sparse": (SkilledLTSFTLinear, False),
}


class SkilledMixin(torch.nn.Module):
    def __init__(
        self,
        model: nn.Module,
        n_tasks: int,
        n_skills: int,
        skilled_variant: str = "learned",
        freeze: bool = True,
        custom_skills: str = None,
        state_dict = None, **kwargs,
    ):
        super().__init__(**kwargs)
        self.underlying_model = model
        self.n_tasks = n_tasks
        self.n_skills = n_skills
        self.skilled_variant = skilled_variant
        self.training = True
        self.add_prior = True

        if freeze:
            for p in self.underlying_model.parameters():
                p.requires_grad = False

        adapter_class, only_attention = VARIANT2CLASS.get(skilled_variant, (SkilledLoRALinear, True))
        self.adapter_class = adapter_class
        skills = self.get_skills(custom_skills)
        replace_layers(self.underlying_model, adapter_class, n_tasks, n_skills, skills, only_attention=only_attention)

        if state_dict is not None:
            self.underlying_model.load_state_dict(state_dict, strict=False)
            self.underlying_model.tie_weights()

    def get_skills(self, custom_skills):
        if self.skilled_variant in ["learned", "hyper", "sparse"]:
            # skills are computed inside each module
            skills = None
        elif self.skilled_variant == "shared":
            skills = torch.ones((self.n_tasks, 1), device=self.device)
        elif self.skilled_variant == "private":
            skills = torch.eye(self.n_tasks, self.n_tasks, device=self.device)
        elif self.skilled_variant == "custom":
            skills = custom_skills
        else:
            raise ValueError

        return skills

    def generate(self, input_ids, *args, **kwargs):
        task_ids = kwargs.pop("task_ids", None)
        tinfo("gen task ids vvvvvvv: %s", task_ids)
        device = self.device
        #task_ids = torch.tensor([0])
        if task_ids != None:
            task_ids = task_ids.long()
            task_ids.to(device)
        inform_layers(self.underlying_model, self.adapter_class, task_ids)
        return self.underlying_model.generate(input_ids=input_ids, *args, **kwargs)

    def forward(self, input_ids, tids=None, *args, **kwargs):
        task_ids = kwargs.pop("task_ids", None)
        task_ids = task_ids.long()
        inform_layers(self.underlying_model, self.adapter_class, task_ids)
        outputs = self.underlying_model.forward(input_ids = input_ids, *args, **kwargs)
        #outputs = super().forward(input_ids, *args, **kwargs)

        if self.training and self.skilled_variant == "learned" and self.add_prior:
            aux_loss = [self.neg_log_IBP(p) for n, p in self.underlying_model.named_parameters() if "skill_logits" in n]
            outputs.loss += torch.stack(aux_loss).sum()

        return outputs

    @staticmethod
    def log_factorial(value):
        return torch.lgamma(value + 1)

    def neg_log_IBP(self, matrix):
        """ Calculate IBP prior contribution - log P(Z)
            Based on https://github.com/davidandrzej/PyIBP/blob/master/PyIBP.py """
        
        # discretise
        N, K = matrix.shape
        matrix = torch.sigmoid(matrix)
        matrix_hard = (matrix > .5).float()
        Z = matrix_hard - matrix.detach() + matrix

        # penalise non-unique histories (columns of Z)
        _, Khs = Z.unique(dim=1, return_counts=True)
        logp = - self.log_factorial(Khs).sum()

        # total feature usage
        m = Z.sum(dim=0)
        m = m[m.nonzero()].squeeze()
        logp += (self.log_factorial(N - m) + self.log_factorial(m - 1)).sum()     

        return - logp


if __name__ == "__main__":
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    tokenizer = T5Tokenizer.from_pretrained("/home/pouramini/pret/t5-base")
    model = T5ForConditionalGeneration.from_pretrained("/home/pouramini/pret/t5-base")
    inputs = ["Tell me, oh Muse, of that ingenious hero who travelled far and wide after he had sacked the famous town of Troy.",
        "Many cities did he visit, and many were the nations with whose manners and customs he was acquainted."]
    inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    task_ids = torch.LongTensor([0, 1])

    for skilled_variant in ["learned", "hyper", "sparse", "shared", "private"]:
        skilled_model = SkilledMixin(model, n_tasks=2, n_skills=2, skilled_variant=skilled_variant)
        logger.warning("forward %s: %s", skilled_variant, skilled_model.forward(task_ids, labels=inputs["input_ids"], add_prior=True, **inputs))
        hyps = skilled_model.generate(task_ids, **inputs)
        hyps = tokenizer.batch_decode(hyps,skip_special_tokens= True)
        logger.warning("generate %s: %s", skilled_variant, hyps) 
