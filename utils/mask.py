import torch
import random

def mask_tensor(input_tensor, mask_indices, mask_type='noise'):
    '''
    将输入张量的指定索引位置的值mask掉
    输入:input_tensor(batch_size, sequence_length, channel, height, width)
    mask_indices: 要mask的索引位置,列表形式,如[4,9,14,19,24,29]
    mask_type: mask的类型:'black'表示mask为黑色,'white'表示mask为白色,'noise'表示mask为随机噪声
    输出:mask后的张量
    注意:为了节省显存没使用clone复制tensor(能否节省存疑),因此会改变原始tensor的值
    '''

    # 确保输入张量的维度正确
    assert input_tensor.dim() == 5 
    
    if mask_type == 'black':
        for idx in mask_indices:
            input_tensor[:, idx, :, :, :] = 1

    elif mask_type == 'white':
        for idx in mask_indices:
            input_tensor[:, idx, :, :, :] = -1

    elif mask_type == 'noise':
        batch_size, sequence_length, channel, height, width = input_tensor.shape
        noise = torch.rand(batch_size, channel, height, width) * 2 - 1
        for idx in mask_indices:
            input_tensor[:, idx, :, :, :] = noise
    
    return input_tensor



def make_mask_indices(sequence_length=30,task='hard'):
    '''
    生成mask的索引位置
    '''
    assert sequence_length > 5
    if task == 'easy':
        # 从4开始，每5帧mask一次
        mask_indices = [i for i in range(4, sequence_length, 5)]
    elif task == 'medium':
        # 每5帧mask一次，但是mask的位置随机
        mask_indices = []
        for i in range(0, sequence_length, 5):
            if i + 5 <= sequence_length:
                mask_indices.append(random.randint(i, i + 4))
    elif task == 'hard':
        mask_indices = random.sample(range(sequence_length), sequence_length//5)
    else:
        raise ValueError('task must be one of easy, medium, hard')
    return mask_indices
