# Dance Frame Extractor

Streamlit app that extracts high-energy frames from dance videos.

## Features
- Upload dance videos (MP4, MOV, AVI, MKV, WEBM)
- Automatically selects 10–12 highest-motion frames
- Gallery view with frame selection
- Download individual frames as JPEG or bulk download as ZIP

## Deployment
Deployed on DigitalOcean droplet as a systemd service with Nginx reverse proxy.

## Tech Stack
- Python / Streamlit
- OpenCV for video processing
- PIL/Pillow for image handling
