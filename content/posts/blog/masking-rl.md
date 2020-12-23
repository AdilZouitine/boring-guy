---
title: "Masking in Deep Reinforcement Learning"
date: 2020-10-31T16:20:06+01:00
math: true
---

# Introduction 
{{< math.inline >}}
<p>
When I started deep reinforcement learning I was faced with an environment where certain actions are not available at every timestep \(t\).
</p>
{{</ math.inline >}}
Naturally a question emerged: **"How can I manage impossible actions?"**.

The first solution I implemented is to assign a negative reward if the agent takes an impossible action.
However, I was not satisfied with this method because it does not explicitly force the agent not to take an impossible action.

Then I decided to use **action masking**.
This method is simple to implement and elegant because it constrains the agent to take only "meaningful" actions. 

Throughout in my practice of deep reinforcement learning I have learned that there are many ways to use masks.
They can be used at any level in the neural network and for different tasks.
Unfortunately there are few mask implementations for Reinforcement Learning available except for this great article by Costa Huang.

The scope of this blog post is to explain the concept of masking and to illustrate it through figures and code.


# Requirements
- A notion of the [Maskovian Decision Processes](https://www.wikiwand.com/en/Markov_decision_process#:~:text=In%20mathematics%2C%20a%20Markov%20decision,control%20of%20a%20decision%20maker.) (MDP)
- Notions of [Policy gradient](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) and [Q-Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) algorithms
- Some knowledge of [Pytorch](https://pytorch.org/) or the basics of [Numpy](https://numpy.org/)


----

# Action level

{{< math.inline >}}
<p>
The primary function of a mask in deep reinforcement learning is to filter out impossible or unavailable actions.
For example Alphastar [1] and Open Ai Five [2] the total number of actions for each time step is \(10^{26}\) and \(1,837,080\).
However, the possible action space for each time step is a small percentage of the available action space. 
The advantage in these applications is double.
The first one is to avoid giving invalid actions to the environment and it is a simple method that helps to manage the huge spaces of action by reducing them consequently. 

</p>
{{</ math.inline >}}


{{< figure library="true" src="/img/masking-rl/action_masking.svg" lightbox="true" >}}
*Figure 1 : Visualisation of an action mask at the logit level*

{{< math.inline >}}
<p>
The idea behind action masking is simple. It consists in replacing the logits associated to impossible actions at \(-\infty\).
</p>
{{</ math.inline >}}
As we represent our values using `float32` we will take the lowest possible value that can be represented with 32 bits.
In `Pytorch` we get this value with the following command:  `torch.finfo(torch.float.dtype).min` and the value is `-3.40e+38`.

**The question now is why applying this mask prevents impossible actions being selected?**

1. **Value-based algorithm (Q-Learning)** :

{{< math.inline >}}
<p>
In the value-based approach, we select the highest estimated value of the action-value function \(Q(s, a)\).
If we apply the action mask, the values associated with the impossible actions will be set to \(-\infty\) so they will never be the highest value and therefore they will never be selected. 
</p>
{{</ math.inline >}}
$$
a = \underset{a \in A}{\operatorname{argmax}} Q(s, . )
$$

2. **Policy Based algorithm (Policy gradient)** :
In the policy-based approach the action is sampled according to the probability distribution at the output of the model.
$$
a \sim \pi_{\theta}(. \mid s)
$$
{{< math.inline >}}
<p>
It is therefore necessary to set the probability associated with the impossible action to 0.
When we apply the mask, the logits associated with the impossible action are at \(-\infty\).
To shift from the logits to the problability domain we use the softmax function.
</p>
{{</ math.inline >}}
$$
Softmax(\vec{z})\_{i} =\frac{e^{z_{i}}}{\sum_{j=1}^{K} e^{z_{j}}} \text { for } i=1, \ldots, K \text { and } \mathbf{z}=\left(z_{1}, \ldots, z_{K}\right) \in \mathbb{R}^{K}
$$
{{< math.inline >}}
<p>
Considering that we have set the value of logits associated with impossible actions to \(-\infty\), the probability of sampling these actions is equal to 0 because \(\lim _{x \rightarrow-\infty} e^{x}=0\).
</p>
{{</ math.inline >}}


Now let's practice and implement action masking for a **discrete** action space and a policy-based algorithm.
For this implementation I was inspired by Costa Huang paper [7].
```python
from typing import Optional

import torch
from torch.distributions.categorical import Categorical
from torch import einsum
from einops import  reduce


class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)
```

```python
logits_or_qvalues = torch.randn((2, 3))
print(logits_or_qvalues)
# tensor([[-1.8222,  1.0769, -0.6567],
#         [-0.6729,  0.1665, -1.7856]])

mask = torch.zeros((2, 3), dtype=torch.bool)
mask[0][2] = True
mask[1][0] = True
mask[1][1] = True
print(mask)
# tensor([[False, False,  True],
#         [ True,  True, False]])

head = CategoricalMasked(logits=logits_or_qvalues)
print(head.probs)
# tensor([[0.0447, 0.8119, 0.1434],
#         [0.2745, 0.6353, 0.0902]])

head_masked = CategoricalMasked(logits=logits_or_qvalues, mask=mask)
print(head_masked.probs)
# tensor([[0.0000, 0.0000, 1.0000],
#         [0.3017, 0.6983, 0.0000]])

print(head.entropy())
# tensor([0.5867, 0.8601])

print(head_masked.entropy())
# tensor([-0.0000, 0.6123])
```
----

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

{{< figure library="true" src="/img/masking-rl/self-attention.svg" lightbox="true" >}}
*Figure 6 :*

$$
\text { Attention }(Q, K, V, Mask)=\operatorname{softmax}\left(\frac{Mask(Q K^{T})}{\sqrt{d_{k}}}\right) V
$$

{{< figure library="true" src="/img/masking-rl/mha.svg" lightbox="true" >}}
*Figure 7 :*





$$
\text { MultiHead }(Q, K, V, Mask)= \operatorname{Concat}(\text { head } {1}, \ldots, \text { head }_{h}) W^{O}
$$

$$
\text { where head }{i} = \text { Attention }(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}, Mask)
$$

```python
import torch
from torch import nn, einsumimport torch.nn.functional as F
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

----

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

----

# Conclusion

----

# Going further

----

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