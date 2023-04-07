# coding=utf8
import sys


def main(item_freq_file):
    dict_file = open( 'his_listen_song.dict', 'w')
    freq_file = open( 'his_listen_song.freq', 'w')

    input_i = 0
    output_i = 0
    with open(item_freq_file, 'r') as fr:
        for l in fr:
            _, vid, uid, freq = l.strip().split(' ')
            gid = (int(uid) & 0xffff000000000000) >> 48
            if gid == 1:
                freq_file.write("{} {}\n".format(output_i, freq))
                dict_file.write("{} {} {}\n".format(uid, vid, output_i))
                output_i += 1
            elif gid == 8:
                dict_file.write("{} {} {}\n".format(uid, vid, input_i))
                input_i += 1


if __name__ == '__main__':
    main(sys.argv[1])
