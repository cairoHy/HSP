import argparse

import numpy


def parse_args():
    parser = argparse.ArgumentParser(description="Shuffle Dataset")

    parser.add_argument("--input", type=str, nargs=5, required=True, help="Dataset")
    parser.add_argument("--suffix", type=str, default="shuf", help="Suffix of Output File")
    parser.add_argument("--seed", default=12345, type=int, help="Random Seed")
    parser.add_argument("--shuffle_anno", action="store_true", help="Shuffle annotation file(s)")

    return parser.parse_args()


def main(args):
    if args.seed:
        numpy.random.seed(args.seed)

    with open(args.input[0], "r", errors='ignore', encoding='utf-8') as frs, \
            open(args.input[1], "r", errors='ignore', encoding='utf-8') as frt:
        data1 = [line for line in frs]
        data2 = [line for line in frt]

    if len(data1) != len(data2):
        raise ValueError("length of files are not equal")

    indices = numpy.arange(len(data1))
    numpy.random.shuffle(indices)

    with open(args.input[0] + "." + args.suffix, "w", encoding='utf-8') as fws:
        with open(args.input[1] + "." + args.suffix, "w", encoding='utf-8') as fwt:
            for idx in indices.tolist():
                fws.write(data1[idx])
                fwt.write(data2[idx])

    if args.shuffle_anno:  # shuffle annotation train file
        anno_tgt_file_name = '.'.join(args.input[2].split('.')[:-1]) + '.' + args.suffix + '.pos'
        with open(args.input[2], "r", errors='ignore', encoding='utf-8') as anno:
            data3 = [line for line in anno]
        if len(data1) != len(data3):
            raise ValueError("length of files are not equal")
        with open(anno_tgt_file_name, "w", encoding='utf-8') as anno_wt:
            for idx in indices.tolist():
                anno_wt.write(data3[idx])

    # shuffle LogicForm train file
    with open(args.input[3], "r", errors='ignore', encoding='utf-8') as lf:
        data4 = [line for line in lf]
    if len(data1) != len(data4):
        raise ValueError("length of files are not equal")
    with open(args.input[3] + '.' + args.suffix, "w", encoding='utf-8') as lf_out:
        for idx in indices.tolist():
            lf_out.write(data4[idx])

    # shuffle sketch train file
    with open(args.input[4], "r", errors='ignore', encoding='utf-8') as sketch_in:
        data5 = [line for line in sketch_in]
    if len(data1) != len(data5):
        raise ValueError("length of files are not equal")
    with open(args.input[4] + '.' + args.suffix, "w", encoding='utf-8') as sketch_out:
        for idx in indices.tolist():
            sketch_out.write(data5[idx])


if __name__ == "__main__":
    main(parse_args())
