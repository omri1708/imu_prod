# imu_repo/realtime/webrtc.py
from __future__ import annotations
import asyncio
from typing import Dict, Any

class ResourceRequired(Exception): ...

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.contrib.media import MediaPlayer, MediaRecorder
    HAS_AIORTC=True
except ImportError:
    HAS_AIORTC=False

class WebRTCSession:
    """Wrapper for WebRTC peer connection."""
    def __init__(self):
        if not HAS_AIORTC:
            raise ResourceRequired("aiortc library required for WebRTC")
        self.pc=RTCPeerConnection()

    async def offer(self)->Dict[str,Any]:
        offer=await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        return {"sdp":self.pc.localDescription.sdp,"type":self.pc.localDescription.type}

    async def answer(self,offer:Dict[str,Any])->Dict[str,Any]:
        desc=RTCSessionDescription(sdp=offer["sdp"],type=offer["type"])
        await self.pc.setRemoteDescription(desc)
        answer=await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return {"sdp":self.pc.localDescription.sdp,"type":self.pc.localDescription.type}

    def add_data_channel(self,label:str):
        return self.pc.createDataChannel(label)

    def add_media(self,kind:str,src:str,dst:str):
        if kind=="audio":
            self.pc.addTrack(MediaPlayer(src).audio)
            return MediaRecorder(dst)
        elif kind=="video":
            self.pc.addTrack(MediaPlayer(src).video)
            return MediaRecorder(dst)
        else:
            raise ValueError("invalid media kind")
