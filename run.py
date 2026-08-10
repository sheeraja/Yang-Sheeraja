import torch

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU Not available，device: CPU")

    tensor = torch.tensor([1.0, 2.0, 3.0])
    tensor = tensor.to(device)
    print(f"tensor: {tensor}")
    print(f"device: {tensor.device}")
    # compute
    result = tensor * 2
    print(result)

if __name__ == "__main__":
    main()

