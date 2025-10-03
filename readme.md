```shell
sudo cog predict -i audio_path=@testing-1-2-3.mp3 -i language=en -i mode=align -i segments="$(cat segments.json)"
```