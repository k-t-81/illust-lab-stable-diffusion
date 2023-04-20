import os

import byte_handler

def main():
    data = b'Hello, this is an example of bytes data that will be split, processed and combined using multiprocessing in Python.'
    temp_dir = 'temp_data'

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    num_parts = byte_handler.split_and_save_data(data, temp_dir)
    combined_data = byte_handler.load_and_combine_data(temp_dir, num_parts)

    print("Original data:", data)
    print("Combined data:", combined_data)
    print("Data is the same:", data == combined_data)

if __name__ == '__main__':
    main()
