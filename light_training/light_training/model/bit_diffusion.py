import torch 

from einops import rearrange, reduce, repeat


BITS = 8
def decimal_to_bits(x, bits = BITS):
    """ expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1 """
    device = x.device

    # x = (x * 255).int().clamp(0, 255)
    x = x.int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device)
    mask = rearrange(mask, 'd -> d 1 1 1')
    x = rearrange(x, 'b c d h w -> b c 1 d h w')

    bits = ((x & mask) != 0).float()
    bits = rearrange(bits, 'b c u d h w -> b (c u) d h w')
    # bits = bits * 2 - 1
    return bits

def bits_to_decimal(x, bits = BITS):
    """ expects bits from -1 to 1, outputs image tensor from 0 to 1 """
    device = x.device

    x = (x > 0).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device, dtype = torch.int32)

    mask = rearrange(mask, 'd -> d 1 1 1')
    x = rearrange(x, 'b (c u) d h w -> b c u d h w', u = bits)
    dec = reduce(x * mask, 'b c u d h w -> b c d h w', 'sum')
    # return (dec / 255).clamp(0., 1.)
    return dec 