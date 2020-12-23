---
title: "Masking in Deep Reinforcement Learning"
date: 2020-10-31T16:20:06+01:00
math: true
---
# TODO
1. Notation math attention
2. Check `CategorialMap`
3. Ecrire feignasse

# Intro 
durant stage à la fin de master en RL un question est venue comment gerer les actions imposible.

Ma première idée était de penaliser l'agent dans la fonction de recompense
pas satisfait j'ai cherché d'autres méthode sur le net (lien stack) 

cette façon est simple
Façon élégante 

Egalement papier il est possible pour les env partiellement observable de masquer au niveau des feature papier open ai

----
Lorsque j'ai commencé l'apprentissage par renforcement profond j'ai été confronté à un environnement où pour certains états il y avait des actions impossible parmi l'espace d'action.
Naturellement un question s'est posé "Comment gérer les actions impossible ?"
La première solution que j'ai implémenté était d'attribuer une récompense négative si l'agent prend une action impossible.
Cependant je n'était pas satisfait de cette solution car elle elle ne contraint pas explicitement l'agent à ne pas prendre d'action impossible.

J'ai donc décidé d'implémenter le masquage d'action.
Cette méthode est simple à implémenter et élégante car elle contraint l'agent à ne prendre que des actions sensée.


Le masquage est utilisé dans de nombreux cas d'application de l'apprentissage par renforcement, tel qu'Alphastar ou Open Ai Five.
Par exemple pour ceux jeux là l'espace d'action disponible à chaque pas de temps $$t$$ est titanesque.
Pour Dota2 il est de 1,837,080 et pour Starcraft II de 10^26 cependant l'espace d'action possible pour chaque pas de temps $t$ est une petite fraction de l'espace d'action disponible. 
L'avantage dans ces cas là est double.
Le premiers est d'eviter de donner des actions invalide à l'environnement  et il va juste ignorer l'action.
Le second est que le masquage est une méthodes simple qui aide à gerer les espaces d'action titanesque.

Egalement dans l'article Hide and seek d'Open ai, les auteurs ont introduit le masquage au niveau de l'extraction de caractéristique pour faire une opération d'auto-attention sur les agents qui sont visible de son point de vue.

Egalement pour optmiser la politique dans un controle de grille (lien papier article mehdi)


Cependant peu d'implémentations existe, l´objectif de cet article est d´expliquer le principe du masquage et montrer qu'il peut intervenir à plusieurs niveaux du réseau.
# Requirements

# Notation

# **Table of contents**:

1. Mask concept
2. Action level
3. Feature level
4. Agent level
5. Conclusion
6. Going further
7. References

----

# Mask principe 
Le principe de masque est simple, c'est une opération  qui permet d'ignorer certains éléments dans un ensemble pour le traitement suivant.
`image mask`


2. Exemple pandas 

# Action level

{{< figure library="true" src="/img/masking-rl/action_masking.svg" lightbox="true" >}}
*Figure 1 :*

$$
\lim _{x \rightarrow-\infty} e^{x}=0
$$

$$
Softmax(\vec{z})\_{i} =\frac{e^{z_{i}}}{\sum_{j=1}^{K} e^{z_{j}}} \text { for } i=1, \ldots, K \text { and } \mathbf{z}=\left(z_{1}, \ldots, z_{K}\right) \in \mathbb{R}^{K}
$$

$$
\underset{a \in A}{\operatorname{argmax}} Q(s_{t}, . )
$$

```python
from typing import Optional

import torch
from torch.distributions.categorical import Categorical


class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.finfo(logits.dtype).max
            logits.masked_fill_(~self.mask, -self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

```

```python
logits_or_qvalues = torch.randn((2, 3))
print(logits_or_qvalues)
# tensor([[ 0.6128, -0.3682,  1.0550],
#         [ 2.3505, -0.0106, -1.2979]])

mask = torch.zeros((2, 3), dtype=torch.bool)
mask[0][2] = True
mask[1][0] = True
mask[1][1] = True
print(mask)

# tensor([[False, False,  True],
#         [ True,  True, False]])
head = CategoricalMasked(logits=logits_or_qvalues)
print(head.probs)
# tensor([[0.3412, 0.1279, 0.5309],
#         [0.8926, 0.0842, 0.0232]])
head_masked = CategoricalMasked(logits=logits_or_qvalues, mask=mask)
print(head_masked.probs)
# tensor([[0.0000, 0.0000, 1.0000],
#         [0.9138, 0.0862, 0.0000]])
```


# Feature level 
{{< figure library="true" src="/img/masking-rl/grid_rl_no_tree.png" lightbox="true" >}}
*Figure 2 :*

{{< figure library="true" src="/img/masking-rl/grid_4_elem_all.svg" lightbox="true" >}}
*Figure 3 :*

```python
# Observation
# Element set => Panda, Watermelon, Scorpion, Dragon
observation = torch.randn(1, 4, 3)
print(observation.size())
# torch.Size([1, 4, 3])  batch size, nb elem set, nb feature
```

{{< figure library="true" src="/img/masking-rl/grid_rl.png" lightbox="true" >}}
*Figure 4 :*

{{< figure library="true" src="/img/masking-rl/grid_4_elem_scorpion_nope.svg" lightbox="true" >}}
*Figure 5 :*

```python
# Mask
mask = torch.ones((1, 4), dtype=torch.bool)
mask[0][2] = False
print(mask)
# tensor([[ True,  True, False,  True]])
print(mask.size())
# torch.Size([1, 4]) # batch size, nb elem set
```

Nous avons maintenant représenté les masques, nous pouvons donc plonger dans le trou du lapin.


{{< figure library="true" src="/img/masking-rl/self-attention.svg" lightbox="true" >}}
*Figure 6 :*

$$
\text { Attention }(Q, K, V, Mask)=\operatorname{softmax}\left(\frac{Mask(Q K^{T})}{\sqrt{d_{k}}}\right) V
$$

{{< figure library="true" src="/img/masking-rl/mha.svg" lightbox="true" >}}
*Figure 7 :*

----



$$
\text { MultiHead }(Q, K, V, Mask)= \operatorname{Concat}(\text { head } {1}, \ldots, \text { head }_{h}) W^{O}
$$

$$
\text { where head }{i} = \text { Attention }(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}, Mask)
$$

```python
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5  # 1/sqrt(dim)
        self.to_qkv = nn.Linear(
            dim, inner_dim * 3, bias=False
        )  # Wq,Wk,Wv for each vector, thats why *3
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.heads

        # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        qkv = self.to_qkv(x)

        # split into multi head attentions
        q, k, v = rearrange(qkv, "b n (h qkv d) -> b h n qkv d", h=h, qkv=3).unbind(
            dim=-2
        )

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if mask is not None:
            mask_value = torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * mask[:, None, None, :]
            dots.masked_fill_(~mask, -mask_value)

        # follow the softmax,q,d,v equation in the paper
        attn = dots.softmax(dim=-1)

        # product of v times whatever inside softmax
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # concat heads into one matrix, ready for next encoder block
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), attn

```

```python
module = MultiHeadAttention(dim=3, heads = 1,  dim_head = 8)
```

```python
# Self attention without mask
output_without_mask, attention_map_without_mask = module(observation)
print(output_without_mask.size())
# torch.Size([1, 4, 3])  batch size, nb elem set, nb feature
print(attention_map_without_mask.size())
# torch.Size([1, 1, 4, 4])  batch size, nb head, nb elem set, nb elem set


# Self attention with mask
output_with_mask, attention_map_with_mask = module(observation, mask)
print(output_with_mask.size())
# torch.Size([1, 4, 3])  batch size, nb elem set, nb feature
print(attention_map_wit_mask.size())
# torch.Size([1, 1, 4, 4])  batch size, nb head, nb elem set, nb elem set

# Equality test
torch.eq(output_without_mask, output_with_mask)
# False
```
{{< load-plotly >}}
{{< plotly json="/files/plotly/masking-rl/attention_without_mask.json" height="450px" >}}
*Figure 8 :*

{{< plotly json="/files/plotly/masking-rl/attention_with_mask.json" height="450px" >}}
*Figure 9 :*

# Agent level

$$
\text{Joint action space : } U=U_{1} \times U_{2} \times \cdots \times U_{n_{t}} \text { with } 1,2, \cdots, n_{t} \text { is the set of agents.}
$$
$$
\text{Possible action for agent } i \text{ : } \mathbf{u}_{t}= \bigcup_{i=1}^{n_{t}} u_{t}^{i} \text{ , with } u \in U
$$
$$
\text{Stochastic joint policy : } \pi\left(\mathbf{u}_{t} \mid s_{t}\right): S \times U \rightarrow[0,1]
$$
$$
\text{Actor-Critic loss function : }\nabla_{\theta} J(\theta)=\mathbb{E}_{s, \mathbf{u}}\left[\nabla_{\theta} \log \pi_{\theta}(\mathbf{u} \mid s) A_{\pi}(s, \mathbf{u})\right]
$$

{{< figure library="true" src="/img/masking-rl/masking_grid.svg" lightbox="true" >}}
*Figure 10 :*

```python
from typing import Optional

import torch
from torch.distributions.categorical import Categorical


class CategoricalMap(Categorical):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):

        self.batch, _, self.height, self.width = logits.size()  # Tuple[int]
        logits = rearrange(logits, "b a h w -> (b h w) a")

        if mask is not None:
            mask = rearrange(mask, "b  h w -> b (h w)")
            self.mask = mask.to(dtype=torch.float32)
        else:
            self.mask = torch.ones(
                (self.batch, self.height * self.width), dtype=torch.float32
            )

        self.nb_agent = reduce(
            self.mask, "b (h w) -> b", "sum", b=self.batch, h=self.height, w=self.width
        )
        super(CategoricalMap, self).__init__(logits=logits)

    def sample(self) -> torch.Tensor:
        action_grid = super().sample()
        action_grid = rearrange(
            action_grid, "(b h w) -> b h w", b=self.batch, h=self.height, w=self.width
        )
        return action_grid

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        action = rearrange(
            action, "b h w -> (b h w)", b=self.batch, h=self.height, w=self.width
        )

        log_prob = super().log_prob(action)
        log_prob = rearrange(
            log_prob, "(b h w) -> b (h w)", b=self.batch, h=self.height, w=self.width
        )
        # Element wise multiplication

        log_prob = einsum("ij,ij->ij", log_prob, self.mask)
        log_prob = reduce(log_prob,  "b (h w) -> b", "sum", b=self.batch, h=self.height, w=self.width
        )
        return log_prob

    def entropy(self) -> torch.Tensor:
        entropy = super().entropy()
        entropy = rearrange(
            entropy, "(b h w) -> b (h w)", b=self.batch, h=self.height, w=self.width
        )
        # Element wise multiplication

        entropy = einsum("ij,ij->ij", entropy, self.mask)

        entropy = reduce(
            entropy, "b (h w) -> b", "sum", b=self.batch, h=self.height, w=self.width
        )

        return entropy / self.nb_agent
```

```python
action_grid_map = torch.randn(1,3, 2, 2)
print(action_grid_map)
# tensor([[[[ 1.0608,  0.4416],
#           [ 1.2075,  0.0888]],

#          [[ 0.1279,  0.0160],
#           [-1.0273,  0.5896]],

#          [[-0.0016,  0.6164],
#           [ 0.1350,  0.5542]]]])
print(action_grid_map.size())
# torch.Size([1, 3, 2, 2]) batch, nb action, height, width

agent_position = torch.tensor([[[True, False],
                               [False, True]]])

print(agent_position)
# tensor([[[ True, False],
#          [False,  True]]])
print(agent_position.size())
# torch.Size([1, 2, 2]) batch, height, width


mass_action_grid = CategoricalMap(logits=action_grid_map)
mass_action_grid_masked = CategoricalMap(logits=action_grid_map, mask=agent_position)

sampled_grid = mass_action_grid.sample()
print(sampled_grid)
# tensor([[[0, 0],
#          [2, 2]]])

sampled_grid_mask = mass_action_grid_masked.sample()
print(sampled_grid_mask)
# tensor([[[1, 1],
#          [2, 1]]])

lp_masked = mass_action_grid_masked.log_prob(sampled_grid)
print(lp_masked)
# tensor([-1.5331]) batch

lp = mass_action_grid.log_prob(sampled_grid)
print(lp)
# tensor([-4.0220]) batch

entropy = mass_action_grid.entropy()
print(entropy)
# tensor([0.9776]) batch

masked_entropy = mass_action_grid_masked.entropy()
print(masked_entropy)
# tensor([1.0256]) batch

```

# Conclusion

# Going further

# References

[1] [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://www.nature.com/articles/s41586-019-1724-z)

[2] [Dota 2 with Large Scale Deep Reinforcement Learning](https://cdn.openai.com/dota-2.pdf)

[3] [Towards Playing Full MOBA Games with Deep Reinforcement Learning](https://papers.nips.cc/paper/2020/file/06d5ae105ea1bea4d800bc96491876e9-Paper.pdf)

[4] [Grid-Wise Control for Multi-Agent Reinforcement Learning in Video Game AI](http://proceedings.mlr.press/v97/han19a/han19a.pdf)

[5] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

[6] [Transformers From Scratch](http://peterbloem.nl/blog/transformers)

[7] [A Closer Look at Invalid Action Masking in Policy Gradient Algorithms](https://arxiv.org/pdf/2006.14171.pdf)

[8] [Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/pdf/1909.07528.pdf)

[9] [Lucidrains github](https://github.com/lucidrains)

[10] [Multi-agents Reinforcement Learning by Mehdi Zouitine](https://mehdi-zouitine.netlify.app/post/2020-07-13-marl/)