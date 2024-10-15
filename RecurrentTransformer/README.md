# Recurrent Transformers for Weak To Strong Generalization

This was an attempt to replicate and perhaps extend results mentioned in the talk at Simons Institute by Tom Goldstein (University of Maryland). The lab tried to achieve weak to strong generalization by stacking neural nets on top of each other with inputs passed in each recurrent step in an additive manner, similar to residual skip connections (He et al).

The interesting ("wild!" as someone in the audience commented) thing was that even stacking transformers just worked for simple problems like addition especially with modifications like Abacus embeddings. The replication for addition mostly worked, but extensions to other domains hasn't shown strong results so far. I agree with Prof. Goldstein's criticism of CoT and thought this was a promising direction to explore, but I definitely need to do further work.

Here is my plan from here for this experiment:
- Perhaps the last first and last couple of layers need to be retrained to work with this paradigm. The addition one I trained from scratch with the truncated gradient propagation method and it worked reasonably well, so this method can't work out of the box for sure. With DeepSeek Math it generates garbled results that are incoherent.
- Maybe this can be integrated with Gated Transformers (see other directory in this repo) for dynamic recurrence and exit. I'm not sure how to tune the gating and get nice gradients from data but we'll see! GDM has this interesting paper that I need to study that may be relevant: https://arxiv.org/pdf/2410.20672
