*A Pokémon bulk scanner using a webcam*

I CoPilot vibe coded python script that uses a webcam, Pokémon cards, OpenCV to detect motion, take a picture, use Azure vision to read text, look for the card code and number, and output that to a usable file.

Coded to get an index over all my scarlet & violet era bulk cards, used for playing. Only works on regulation mark G and higher.

I use this box from Clas Ohlson https://www.clasohlson.com/no/SmartStore-innsats-for-Home-og-Classic-4-og-14/p/44-5238-7

And a Trust 23637 Webcam on my M3 Mac

*Setup*
Uses python3 and pip3
# setup env: python3 -m venv <path>/cardScan
# env: source <path>/cardScan/bin/activate
# pip3 install azure-ai-vision-imageanalysis  
# pip3 install opencv-python
# export VISION_KEY=<your_key>
# export VISION_ENDPOINT=<your endpoint>