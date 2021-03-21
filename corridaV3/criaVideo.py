#https://github.com/cdemutiis/Behavioral-Cloning/blob/master/video.py
from moviepy.editor import ImageSequenceClip
import argparse

def main():
    parser = argparse.ArgumentParser(description='Criando video.')
    parser.add_argument(
        'pasta_imagem',
        type=str,
        default='',
        help='Pasta da imagem. O video é criado destas imagens.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames por segundo)')
    args = parser.parse_args()

    video_file = args.image_folder + '.mp4'
    print("Criando video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(args.image_folder, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()