    class ScreenSharing {
        constructor() {
          this.peerConnections = {};
          this.localStreams = {};
          this.makingOffer = {};
          this.ignoreOffer = {};
          this.polite = {};
          this.peerData = {};
          this.shareSockets = {};
          this.messageQueue = {};
          this.rtcConfig = {
            iceServers: [
              {
                urls: "turn:relay1.expressturn.com:3478",
                username: "ef0OSL1KJJM05XK829",
                credential: "4HQgeo0mAcWomtgj",
              },
              {
                urls: [
                  "stun:stun.l.google.com:19302",
                  "stun:stun1.l.google.com:19302",
                  "stun:stun2.l.google.com:19302",
                ],
              },
            ],
            iceTransportPolicy: "all",
            bundlePolicy: "max-bundle",
            rtcpMuxPolicy: "require",
            iceCandidatePoolSize: 10,
          };
        }

        async initScreenSharing(chatPanelId, sender, recipient) {
          try {
            console.log(`Initializing screen sharing for ${chatPanelId}`);
            this.cleanupScreenSharing(chatPanelId);
            this.peerData[chatPanelId] = {
              sender: sender,
              recipient: recipient,
            };
            this.polite[chatPanelId] = true;
            this.makingOffer[chatPanelId] = false;
            this.ignoreOffer[chatPanelId] = false;

            this.shareSockets[chatPanelId] = new WebSocket(
              window.location.protocol === "https:"
                ? `wss://${window.location.host}/ws/screenshare/${sender}/${recipient}/`
                : `ws://${window.location.host}/ws/screenshare/${sender}/${recipient}/`
            );
            this.shareSockets[chatPanelId].open = async () => {
              if (!this.peerConnections[chatPanelId]) {
                await this.createPeerConnection(chatPanelId);
              }
            };
            this.shareSockets[chatPanelId].sender = sender;
            this.shareSockets[chatPanelId].recipient = recipient;
            this.setupSocketHandlers(chatPanelId);
            this.createVideoElements(chatPanelId);
            this.setupEventListeners(chatPanelId, sender, recipient);

            console.log(`Screen sharing initialized for ${chatPanelId}`);
          } catch (error) {
            console.error(`Error initializing screen sharing: ${error}`);
            this.showError(chatPanelId, "Failed to initialize screen sharing");
          }
        }

        setupSocketHandlers(chatPanelId) {
          const socket = this.shareSockets[chatPanelId];

          socket.onopen = async () => {
            console.log(`Screen share WebSocket opened for ${chatPanelId}`);
            this.updateConnectionStatus(chatPanelId, "connecting");
            await this.createPeerConnection(chatPanelId);
            this.processMessageQueue(chatPanelId);
          };

          socket.onerror = (error) => {
            console.error(`WebSocket error for ${chatPanelId}:`, error);
            this.updateConnectionStatus(chatPanelId, "disconnected");
          };

          socket.onclose = () => {
            console.log(`WebSocket closed for ${chatPanelId}`);
            this.updateConnectionStatus(chatPanelId, "disconnected");
            this.cleanupSocket(chatPanelId);
          };

          socket.onmessage = async (event) => {
            try {
              const data = JSON.parse(event.data);
              console.log(`Received ${data.type} for ${chatPanelId}`, data);

              if (!this.isValidMessageType(data.type)) {
                console.warn(`Invalid message type: ${data.type}`);
                return;
              }

              if (!this.isPeerConnectionReady(chatPanelId)) {
                console.log(`PeerConnection not ready, queuing ${data.type}`);
                this.queueMessage(chatPanelId, data);
                return;
              }

              await this.processSignalingMessage(chatPanelId, data);
            } catch (error) {
              console.error(
                `Error handling message for ${chatPanelId}:`,
                error
              );
            }
          };
        }

        createVideoElements(chatPanelId) {
          const container = document.getElementById(
            `screen-share-container-${chatPanelId}`
          );
          let remoteVideo = document.getElementById(
            `remoteVideo-${chatPanelId}`
          );
          if (remoteVideo) {
            remoteVideo.muted = true;
            remoteVideo.playsInline = true;

            const playPromise = remoteVideo.play();
            if (playPromise !== undefined) {
              playPromise.catch(() => {
                this.showPlayButton(chatPanelId, remoteVideo);
              });
            }
          }
          if (!container) {
            console.error(`Container not found for ${chatPanelId}`);
            return;
          }
          if (!remoteVideo) {
            remoteVideo = document.createElement("video");
            remoteVideo.id = `remoteVideo-${chatPanelId}`;
            remoteVideo.autoplay = true;
            remoteVideo.muted = true; // Required for autoplay
            remoteVideo.playsInline = true;
            container.appendChild(remoteVideo);
          }
          remoteVideo.onplay = () => {
            const playBtn = container.querySelector(".video-play-button");
            if (playBtn) playBtn.remove();
          };

          if (!document.getElementById(`localVideo-${chatPanelId}`)) {
            const localVideo = document.createElement("video");
            localVideo.id = `localVideo-${chatPanelId}`;
            localVideo.className = "localVideo";
            localVideo.autoplay = true;
            localVideo.muted = true;
            localVideo.playsInline = true;
            container.appendChild(localVideo);
          }

          if (!document.getElementById(`remoteVideo-${chatPanelId}`)) {
            const remoteVideo = document.createElement("video");
            remoteVideo.id = `remoteVideo-${chatPanelId}`;
            remoteVideo.className = "remoteVideo";
            remoteVideo.autoplay = true;
            remoteVideo.muted = true;
            remoteVideo.playsInline = true;
            container.appendChild(remoteVideo);
          }

          if (!document.getElementById(`connection-status-${chatPanelId}`)) {
            const statusIndicator = document.createElement("div");
            statusIndicator.id = `connection-status-${chatPanelId}`;
            statusIndicator.className = "connection-status";
            container.appendChild(statusIndicator);
          }
        }
        showPlayButton(chatPanelId) {
          const container = document.getElementById(
            `screen-share-container-${chatPanelId}`
          );
          const video = document.getElementById(`remoteVideo-${chatPanelId}`);
          if (!container || !video) return;
          const existingBtn = container.querySelector(".video-play-button");
          if (existingBtn) existingBtn.remove();

          const playBtn = document.createElement("button");
          playBtn.className = "video-play-button";
          playBtn.textContent = "Play Video";
          playBtn.onclick = () => {
            video.play().catch((e) => console.error("Playback failed:", e));
          };
          container.appendChild(playBtn);
        }
        setupEventListeners(chatPanelId, sender, recipient) {
          $(`#shareBtn-${chatPanelId}`)
            .off("click")
            .on("click", async () => {
              if (this.localStreams[chatPanelId]) {
                await this.stopScreenShare(chatPanelId);
              } else {
                await this.startScreenShare(chatPanelId, sender, recipient);
              }
            });

          $(document).on(
            `click`,
            `.close-chat-content[data-id="${chatPanelId}"]`,
            () => {
              this.cleanupScreenSharing(chatPanelId);
            }
          );
        }

        async startScreenShare(chatPanelId, sender, recipient) {
          try {
            console.log(`Starting screen share for ${chatPanelId}`);
            $(`#screen-share-container-${chatPanelId}`).show();
            this.updateConnectionStatus(chatPanelId, "connecting");
            if (!(await this.createPeerConnection(chatPanelId))) {
              throw new Error("Failed to create peer connection");
            }
            await this.createPeerConnection(chatPanelId);
            this.localStreams[chatPanelId] =
              await navigator.mediaDevices.getDisplayMedia({
                video: {
                  displaySurface: "monitor",
                },
                audio: false,
              });
            const localVideo = document.getElementById(
              `localVideo-${chatPanelId}`
            );
            if (localVideo) {
              localVideo.srcObject = this.localStreams[chatPanelId];
              localVideo.classList.add("active");
              localVideo
                .play()
                .catch((e) => console.log("Local video play error:", e));
            }
            this.localStreams[chatPanelId].getTracks().forEach((track) => {
              this.peerConnections[chatPanelId].addTrack(
                track,
                this.localStreams[chatPanelId]
              );
              console.log(`Added ${track.kind} track to peer connection`);
              track.onended = () => {
                console.log("Screen sharing track ended");
                this.stopScreenShare(chatPanelId);
              };
            });
            this.localStreams[chatPanelId].getVideoTracks()[0].onended = () => {
              console.log("User stopped screen sharing");
              this.stopScreenShare(chatPanelId);
            };

            console.log(`Screen sharing started for ${chatPanelId}`);
          } catch (error) {
            console.error(`Error starting screen share: ${error}`);
            this.stopScreenShare(chatPanelId);
            if (error.name !== "NotAllowedError") {
              this.showError(
                chatPanelId,
                `Could not start screen sharing: ${error.message}`
              );
            }
          }
        }

        async createPeerConnection(chatPanelId) {
          if (this.peerConnections[chatPanelId]) {
            this.peerConnections[chatPanelId].close();
            delete this.peerConnections[chatPanelId];
          }

          try {
            console.log(`Creating PeerConnection for ${chatPanelId}`);
            if (this.peerConnections[chatPanelId]) {
              this.peerConnections[chatPanelId].close();
            }
            this.peerConnections[chatPanelId] = new RTCPeerConnection(
              this.rtcConfig
            );
            this.updateConnectionStatus(chatPanelId, "connecting");
            this.setupPeerConnectionHandlers(chatPanelId);
            console.log(`PeerConnection created for ${chatPanelId}`);
            return true;
          } catch (error) {
            console.error(`Error creating PeerConnection: ${error}`);
            this.updateConnectionStatus(chatPanelId, "disconnected");
            return false;
          }
        }

        setupPeerConnectionHandlers(chatPanelId) {
          const pc = this.peerConnections[chatPanelId];
          if (!pc) return;
          const [_, sender, recipient] =
            chatPanelId.match(/chat-panel-(.+?)-(\d+)/) || [];
          pc.onicecandidate = ({ candidate }) => {
            console.log(`ICE candidate for ${chatPanelId}:`, candidate);
            if (
              candidate &&
              this.shareSockets[chatPanelId]?.readyState === WebSocket.OPEN
            ) {
              this.shareSockets[chatPanelId].send(
                JSON.stringify({
                  type: "candidate",
                  candidate: candidate,
                })
              );
            }
          };

          pc.onnegotiationneeded = async () => {
            try {
              console.log(`Negotiation needed for ${chatPanelId}`);
              this.makingOffer[chatPanelId] = true;
              await pc.setLocalDescription();
              console.log(`Local description set:`, pc.localDescription);

              if (
                this.shareSockets[chatPanelId]?.readyState === WebSocket.OPEN
              ) {
                this.shareSockets[chatPanelId].send(
                  JSON.stringify({
                    type: "offer",
                    offer: pc.localDescription,
                  })
                );
                console.log(`Offer sent for ${chatPanelId}`);
              }
            } catch (err) {
              console.error(`Negotiation error for ${chatPanelId}:`, err);
            } finally {
              this.makingOffer[chatPanelId] = false;
            }
          };

          pc.oniceconnectionstatechange = () => {
            const state = pc.iceConnectionState;
            console.log(
              `ICE connection state changed to ${state} for ${chatPanelId}`
            );
            this.updateConnectionStatus(chatPanelId, state);
            if (pc.iceConnectionState === "failed") {
              pc.restartIce();
            }
            if (state === "failed") {
              console.log(`Restarting ICE for ${chatPanelId}`);
              pc.restartIce();
            }
          };
          pc.ontrack = (event) => {
            console.log("Track event received:", event);
            const remoteVideo = document.getElementById(
              `remoteVideo-${chatPanelId}`
            );
            if (!remoteVideo) {
              console.error("Remote video element not found");
              return;
            }
            if (remoteVideo.srcObject) {
              remoteVideo.srcObject
                .getTracks()
                .forEach((track) => track.stop());
            }
            remoteVideo.playsInline = true;
            remoteVideo.muted = true;

            if (remoteVideo.srcObject) {
              remoteVideo.srcObject
                .getTracks()
                .forEach((track) => track.stop());
            }
            if (event.streams && event.streams.length > 0) {
              console.log(
                "Attaching remote stream with tracks:",
                event.streams[0].getTracks().map((t) => t.kind)
              );

              remoteVideo.srcObject = event.streams[0];
              remoteVideo.playsInline = true;
              remoteVideo.muted = true;
              const playPromise = remoteVideo.play();
              if (playPromise !== undefined) {
                playPromise.catch((error) => {
                  console.log("Autoplay prevented, showing play button");
                  this.showPlayButton(chatPanelId, remoteVideo);
                });
              }
              $(`#screen-share-container-${chatPanelId}`).show();
              this.ensureVideoPlayback(remoteVideo, chatPanelId);
            }
          };
          pc.onicegatheringstatechange = () => {
            console.log(
              `ICE gathering state for ${chatPanelId}: ${pc.iceGatheringState}`
            );
          };

          pc.onsignalingstatechange = () => {
            console.log(
              `Signaling state for ${chatPanelId}: ${pc.signalingState}`
            );
          };

          pc.onconnectionstatechange = () => {
            const state = pc.connectionState;
            console.log(
              `Connection state changed to ${state} for ${chatPanelId}`
            );

            switch (state) {
              case "connected":
                this.handleConnectionSuccess(chatPanelId);
                break;
              case "failed":
                setTimeout(() => {
                  if (pc.iceConnectionState !== "closed") {
                    pc.restartIce();
                  }
                }, 2000);
                break;
              case "disconnected":
                this.cleanupScreenSharing(chatPanelId);
                break;
            }
          };
        }
        handleConnectionSuccess(chatPanelId) {
        console.log(`Connection established for ${chatPanelId}`);
        const statusElement = document.getElementById(`connection-status-${chatPanelId}`);
        if (statusElement) {
            statusElement.className = "connection-status connected";
          }
        this.monitorConnectionQuality(chatPanelId);
        }

        monitorConnectionQuality(chatPanelId) {
          const pc = this.peerConnections[chatPanelId];
          if (!pc) return;
            setInterval(() => {
              pc.getStats().then(stats => {
                stats.forEach(report => {
                    if (report.type === 'candidate-pair' && report.nominated) {
                        console.log(`Connection quality for ${chatPanelId}:`, {
                            rtt: report.currentRoundTripTime,
                            availableOutgoingBitrate: report.availableOutgoingBitrate
                        });
                    }
                });
            });
          }, 10000);
        }
        handleConnectionFailure(chatPanelId) {
          this.cleanupScreenSharing(chatPanelId);

          const { sender, recipient } = this.peerData[chatPanelId] || {};
          if (!sender || !recipient) return;
          if (sender && recipient) {
            setTimeout(() => {
              this.initScreenSharing(chatPanelId, sender, recipient);
            }, 2000);
          }
        }
        ensureVideoPlayback(videoElement, chatPanelId, attempt = 0) {
          if (attempt > 3) {
            console.error("Max playback attempts reached");
            this.showPlayButton(chatPanelId, videoElement);
            return;
          }

          videoElement
            .play()
            .then(() => {
              console.log("Video playback started successfully");
              videoElement.classList.add("active");
            })
            .catch((error) => {
              console.warn(`Playback attempt ${attempt + 1} failed:`, error);
              if (error.name === "NotAllowedError") {
                this.showPlayButton(chatPanelId, videoElement);
              } else {
                setTimeout(() => {
                  this.ensureVideoPlayback(
                    videoElement,
                    chatPanelId,
                    attempt + 1
                  );
                }, 500);
              }
            });
        }
        handleVideoPlayback(videoElement, chatPanelId) {
          videoElement.muted = true;
          const playPromise = videoElement.play();
          if (playPromise !== undefined) {
            playPromise
              .catch((error) => {
                console.log("Autoplay prevented, showing play button");
                this.showPlayButton(chatPanelId, videoElement);
              })
              .then(() => {
                console.log("Video playback started successfully");
              });
          }
        }

        showPlayButton(chatPanelId, videoElement) {
          const container = document.getElementById(
            `screen-share-container-${chatPanelId}`
          );
          if (!container) return;
          const existingBtn = container.querySelector(".video-play-button");
          if (existingBtn) existingBtn.remove();
          const playBtn = document.createElement("button");
          playBtn.className = "video-play-button btn btn-primary";
          playBtn.textContent = "Play Video";
          playBtn.onclick = () => {
            videoElement
              .play()
              .then(() => {
                playBtn.remove();
                console.log("Playback started after user interaction");
              })
              .catch((e) => {
                console.error("Playback failed:", e);
                this.showError(chatPanelId, "Could not start video playback");
              });
          };

          container.appendChild(playBtn);
        }

        async processSignalingMessage(chatPanelId, data) {
          if (!this.peerConnections[chatPanelId]) {
            console.log(`PeerConnection not ready, queuing ${data.type}`);
            this.queueMessage(chatPanelId, data);
            return;
          }

          const pc = this.peerConnections[chatPanelId];
          const currentDescription =
            pc.remoteDescription || pc.localDescription;

          if (currentDescription && data.type === currentDescription.type) {
            console.log(`Ignoring duplicate ${data.type}`);
            return;
          }
          switch (data.type) {
            case "offer":
              await this.handleOffer(chatPanelId, data.offer);
              break;
            case "answer":
              await this.handleAnswer(chatPanelId, data.answer);
              break;
            case "candidate":
              await this.handleCandidate(chatPanelId, data.candidate);
              break;
            case "stop_screen_share":
              await this.stopScreenShare(chatPanelId);
              break;
          }
        }
        async handleOffer(chatPanelId, offer) {
          try {
            const pc = this.peerConnections[chatPanelId];
            if (!pc) return;

            const offerCollision =
              this.makingOffer[chatPanelId] ||
              (pc.signalingState !== "stable" &&
                pc.signalingState !== "have-local-offer");

            this.ignoreOffer[chatPanelId] =
              !this.polite[chatPanelId] && offerCollision;
            if (this.ignoreOffer[chatPanelId]) {
              console.log(`Ignoring offer for ${chatPanelId} due to collision`);
              return;
            }
            this.messageQueue[chatPanelId] = [];

            await pc.setRemoteDescription(new RTCSessionDescription(offer));

            if (offer.type === "offer") {
              await pc.setLocalDescription();
              this.shareSockets[chatPanelId].send(
                JSON.stringify({
                  type: "answer",
                  answer: pc.localDescription,
                })
              );
            }
          } catch (error) {
            console.error(`Error handling offer: ${error}`);
            this.cleanupScreenSharing(chatPanelId);
          }
        }
        async handleAnswer(chatPanelId, answer) {
          try {
            console.log("Processing answer", answer);
            const pc = this.peerConnections[chatPanelId];

            if (!pc || pc.signalingState === "closed") {
              console.warn("PeerConnection not ready for answer");
              return;
            }

            if (
              pc.signalingState === "have-local-offer" ||
              pc.signalingState === "stable"
            ) {
              if (!pc.remoteDescription) {
                await pc.setRemoteDescription(
                  new RTCSessionDescription(answer)
                );
                console.log("Answer processed successfully");
              } else {
                console.log("Ignoring duplicate answer");
              }
            } else {
              console.warn(
                `Unexpected signaling state for answer: ${pc.signalingState}`
              );
              this.queueMessage(chatPanelId, { type: "answer", answer });
            }
          } catch (error) {
            console.error("Error handling answer:", error);
            throw error;
          }
        }
        async handleCandidate(chatPanelId, candidate) {
          try {
            const pc = this.peerConnections[chatPanelId];
            if (!pc || pc.signalingState === "closed") return;
            if (!candidate || this.ignoreOffer[chatPanelId]) return;
            if (
              candidate.candidate.includes("typ host") &&
              pc.remoteDescription &&
              pc.remoteDescription.type === "answer"
            ) {
              console.log("Skipping host candidate after answer");
              return;
            }
            await pc.addIceCandidate(new RTCIceCandidate(candidate));
          } catch (err) {
            if (!this.ignoreOffer[chatPanelId]) {
              console.error("Failed to add ICE candidate:", err);
            }
          }
        }
        async stopScreenShare(chatPanelId) {
          try {
            console.log(`Stopping screen share for ${chatPanelId}`);
            if (this.localStreams[chatPanelId]) {
              this.localStreams[chatPanelId]
                .getTracks()
                .forEach((track) => track.stop());
              delete this.localStreams[chatPanelId];
            }
            const localVideo = document.getElementById(
              `localVideo-${chatPanelId}`
            );
            const remoteVideo = document.getElementById(
              `remoteVideo-${chatPanelId}`
            );
            if (localVideo) {
              localVideo.srcObject = null;
              localVideo.classList.remove("active");
            }
            if (remoteVideo) {
              remoteVideo.srcObject = null;
              remoteVideo.classList.remove("active");
            }
            if (this.peerConnections[chatPanelId]) {
              this.peerConnections[chatPanelId].close();
              delete this.peerConnections[chatPanelId];
            }
            $(`#screen-share-container-${chatPanelId}`).hide();
            this.updateConnectionStatus(chatPanelId, "disconnected");
            if (this.shareSockets[chatPanelId]?.readyState === WebSocket.OPEN) {
              this.shareSockets[chatPanelId].send(
                JSON.stringify({
                  type: "stop_screen_share",
                })
              );
            }

            console.log(`Screen sharing stopped for ${chatPanelId}`);
          } catch (error) {
            console.error(`Error stopping screen share: ${error}`);
          }
        }

        cleanupScreenSharing(chatPanelId) {
          if (this.shareSockets[chatPanelId]) {
            this.shareSockets[chatPanelId].close();
            delete this.shareSockets[chatPanelId];
          }
          if (this.peerConnections[chatPanelId]) {
            this.peerConnections[chatPanelId].close();
            delete this.peerConnections[chatPanelId];
          }
          if (this.localStreams[chatPanelId]) {
            this.localStreams[chatPanelId]
              .getTracks()
              .forEach((track) => track.stop());
            delete this.localStreams[chatPanelId];
          }
          const container = document.getElementById(
            `screen-share-container-${chatPanelId}`
          );
          if (container) {
            container.style.display = "none";
          }
        }

        cleanupSocket(chatPanelId) {
          if (this.shareSockets[chatPanelId]) {
            this.shareSockets[chatPanelId].close();
            delete this.shareSockets[chatPanelId];
          }
        }

        async processMessageQueue(chatPanelId) {
          if (
            !this.messageQueue[chatPanelId] ||
            this.messageQueue[chatPanelId].length === 0
          ) {
            return;
          }

          console.log(
            `Processing ${this.messageQueue[chatPanelId].length} queued messages for ${chatPanelId}`
          );

          while (
            this.messageQueue[chatPanelId].length > 0 &&
            this.isPeerConnectionReady(chatPanelId)
          ) {
            const message = this.messageQueue[chatPanelId].shift();
            try {
              await this.processSignalingMessage(chatPanelId, message);
            } catch (error) {
              console.error(`Error processing queued ${message.type}:`, error);
            }
          }
        }

        queueMessage(chatPanelId, message) {
          if (!this.messageQueue[chatPanelId]) {
            this.messageQueue[chatPanelId] = [];
          }
          this.messageQueue[chatPanelId].push(message);
          console.log(`Message queued for ${chatPanelId}:`, message.type);
        }

        updateConnectionStatus(chatPanelId, state) {
          const element = document.getElementById(
            `connection-status-${chatPanelId}`
          );
          if (!element) {
            console.warn(`Status element not found for ${chatPanelId}`);
            return;
          }

          element.className = "connection-status";
          const stateMap = {
            connected: "connected",
            completed: "connected",
            checking: "connecting",
            connecting: "connecting",
            disconnected: "disconnected",
            failed: "disconnected",
            closed: "disconnected",
          };

          const statusClass = stateMap[state] || "disconnected";
          element.classList.add(statusClass);
          element.title = `Status: ${state}`;
        }

        showError(chatPanelId, message) {
          console.error(`Error for ${chatPanelId}: ${message}`);
          alert(`Screen Sharing Error: ${message}`);
        }

        isValidMessageType(type) {
          const validTypes = [
            "offer",
            "answer",
            "candidate",
            "stop_screen_share",
          ];
          return validTypes.includes(type);
        }

        isPeerConnectionReady(chatPanelId) {
          return (
            this.peerConnections[chatPanelId] &&
            this.peerConnections[chatPanelId].signalingState !== "closed"
          );
        }
      }

      const screenSharing = new ScreenSharing();

      var onlineSocket = new WebSocket(
        "ws://" + window.location.host + "/ws/online/"
      );
      onlineSocket.onmessage = function (event) {
        var data = JSON.parse(event.data);
        if (data.type === "update_online_users") {
          updateOnlineUserList(data.online_users);
        }
      };
      function updateOnlineUserList(onlineUsers) {
        document.querySelectorAll(".user-status").forEach((element) => {
          element.classList.remove("online");
          element.classList.add("offline");
        });
        onlineUsers.forEach((user) => {
          let userElement = document.getElementById(
            "user-status-" + user.user__id
          );
          if (userElement) {
            userElement.classList.remove("offline");
            userElement.classList.add("online");
          }
        });
      }
      document.querySelector(".menu-toggle").addEventListener("click", () => {
        document.querySelector(".nav-menu").classList.toggle("active");
      });
      $(document).ready(function () {
        let chatPanels = [];
        $(".chat-button").click(function () {
          $(".chat-panel").toggleClass("active");
        });
        $(".close-chat").click(function () {
          $(".chat-panel").removeClass("active");
        });
        $(".chat-list li").click(function () {
          let username = $(this).data("username");
          let userId = $(this).data("id");
          let chatPanelId = "chat-panel-" + username + "-" + userId;
          let sender = "{{ request.user.username }}";
          let recipient = username;
          let chatMessages = "${chat-messages}";
          let existingPanel = $("#" + chatPanelId);
          if (existingPanel.length) {
            existingPanel.addClass("active");
            return;
          }
          console.log("Chat Panel", chatPanelId);
          loadOldMessages(sender, recipient, chatMessages, chatPanelId);

          let chatPanel = $(`
<div class="chat-content-panel" id="${chatPanelId}">
    <div class="chat-header" style="font-family: serif;">
        ${username}
        <div>
            <button class="maximize-chat" data-id="${chatPanelId}">□</button>
            <button class="close-chat-content" data-id="${chatPanelId}">&times;</button>
        </div>
    </div>
    <div class="chat-container">
        <div class="container">
            <div class="chat-container2">
                <div class="messages" id="chat-messages-${chatPanelId}" class="chat-messages"></div>
                <div class="screen-share-container" id="screen-share-container-${chatPanelId}" style="display: none;">
                    <video id="localVideo-${chatPanelId}" class="localVideo" autoplay muted></video>
                    <video id="remoteVideo-${chatPanelId}" class="remoteVideo" autoplay></video>
                </div>
            </div>
            <div class="chat-input" id="chat-input">
                <input autocomplete="off" type="text" class="message-input" id="message-input-${chatPanelId}" placeholder="Type your message...">
                <button id="send-btn-${chatPanelId}" class="send-btn">
                    <img src="https://img.icons8.com/?size=100&id=2837&format=png&color=000000" alt="send">
                </button>
                <button id="shareBtn-${chatPanelId}" class="shareBtn">
                    <img src="https://img.icons8.com/?size=100&id=IId2iYTrumrQ&format=png&color=000000" alt="share">
                </button>
            </div>
        </div>
    </div>
</div>
`);
          $("body").append(chatPanel);
          chatPanels.push(chatPanelId);
          $(`#${chatPanelId}`).addClass("active");
          console.log("Initializing chat WebSocket...");
          console.log(sender, recipient);

          bringPanelToFront(chatPanelId);

          let chatSocket = new WebSocket(
            `ws://${window.location.host}/ws/chat/${sender}/${recipient}/`
          );

          $(document).on("click", `#send-btn-${chatPanelId}`, function () {
            sendMessage(chatSocket, chatPanelId);
          });
          $(`#message-input-${chatPanelId}`).keypress(function (event) {
            if (event.key === "Enter") sendMessage(chatSocket, chatPanelId);
          });

          function sendMessage(chatSocket, chatPanelId) {
            const messageInput = $(`#message-input-${chatPanelId}`);
            const message = messageInput.val().trim();
            if (!message) return;
            chatSocket.send(JSON.stringify({ message: message }));
            messageInput.val("");
          }
          function updateChatPanelPositions() {
            let startRight = 310;
            let panelWidth = 300;
            let gap = 8;

            for (let i = chatPanels.length - 1; i >= 0; i--) {
              let panelId = chatPanels[i];
              $("#" + panelId).css({
                right: startRight + "px",
                display: "block",
              });
              startRight += panelWidth + gap;
            }
          }

          $(document).on("click", ".close-chat-content", function () {
            let panelId = $(this).data("id");
            $("#" + panelId).remove();
            chatPanels = chatPanels.filter((id) => id !== panelId);
            updateChatPanelPositions();
          });
          function bringPanelToFront(panelId) {
            let panel = $("#" + panelId);
            panel.css(
              "z-index",
              Math.max(...chatPanels.map((id) => $("#" + id).css("z-index"))) +
                1
            );
          }

          chatSocket.onmessage = function (event) {
            const data = JSON.parse(event.data);
            console.log("Received message:", data);
            renderMessage(data, chatPanelId);
          };
    $(document).on('click', '.maximize-chat', function() {
    const panelId = $(this).data('id');
    const panel = $(`#${panelId}`);
    panel.toggleClass('maximized');
    const button = $(this);
    if (panel.hasClass('maximized')) {
        button.text('❐');
    } else {
        button.text('□');
    }
    if (panel.hasClass('maximized')) {
        panel.css('z-index', '1100');
    } else {
        panel.css('z-index', '1030');
    }
    if (panel.hasClass('maximized')) {
        $(`#screen-share-container-${panelId}`).css('height', '50%');
    } else {
        $(`#screen-share-container-${panelId}`).css('height', 'auto');
    }
});

          function renderMessage(data, chatPanelId) {
            const chatMessages = $(`#chat-messages-${chatPanelId}`);
            let chatPanel = $("#" + chatPanelId);
            const newMessage = `
        <div class="message-container ${
          data.sender === "{{ request.user.username }}" ? "me" : "other"
        }">
            <div class="message-bubble">
                <div class="username">${data.sender}</div>
                <div class="message-content">${data.message || ""}</div>
                <div class="message-timestamp">${new Date().toLocaleTimeString()}</div>
            </div>
        </div>`;
            chatMessages.append(newMessage);
            chatMessages.scrollTop(chatMessages.prop("scrollHeight"));
          }
          updateChatPanelPositions();
          markMessagesAsRead(username);
        });

        function loadOldMessages(sender, recipient, chatMessages, chatPanelId) {
          $.ajax({
            url: `/load-messages/${sender}/${recipient}/`,
            type: "GET",
            success: function (data) {
              const chatMessages = $(
                `#${chatPanelId} #chat-messages-${chatPanelId}`
              );
              data.messages.forEach((msg) => {
                const newMessage = `
                        <div class="message-container ${
                          msg.sender === sender ? "me" : "other"
                        }">
                            <div class="message-bubble">
                                <div class="username">${msg.sender}</div>
                                <div class="message-content">${
                                  msg.content
                                }</div>
                                <div class="message-timestamp">${
                                  msg.timestamp
                                }</div>
                            </div>
                        </div>`;
                chatMessages.append(newMessage);
              });
              chatMessages.scrollTop(chatMessages.prop("scrollHeight"));
            },
            error: function (error) {
              console.error("Error loading messages:", error);
            },
          });
        }

        $(document).on("click", ".close-chat-content", function () {
          let panelId = $(this).data("id");
          $("#" + panelId).remove();
          chatPanels = chatPanels.filter((id) => id !== panelId);
        });

        var notificationSocket = new WebSocket(
          "ws://" + window.location.host + "/ws/notifications/"
        );
        notificationSocket.onmessage = function (event) {
          var data = JSON.parse(event.data);
          if (data.type === "unread_count") {
            updateUnreadCounts(data.unread_counts);
          }
        };

        function markMessagesAsRead(username) {
          $.ajax({
            url: "/mark-as-read/",
            type: "POST",
            data: {
              username: username,
              csrfmiddlewaretoken: "{{ csrf_token }}",
            },
            success: function (response) {
              $("#notification-badge-" + username).hide();
              $("#chat-notification").hide();
            },
          });
        }
        $(".chat-list li").click(function () {
          const username = $(this).data("username");
          const userId = $(this).data("id");
          const chatPanelId = `chat-panel-${username}-${userId}`;
          const sender = "{{ request.user.username }}";
          const recipient = username;
          if (!document.getElementById(`connection-status-${chatPanelId}`)) {
            $(`#${chatPanelId} .chat-header`).append(
              `<div class="connection-status" id="connection-status-${chatPanelId}"></div>`
            );
          }
          screenSharing.initScreenSharing(chatPanelId, sender, recipient);
        });
        function updateUnreadCounts(unreadCounts) {
          let userItems = document.querySelectorAll(".chat-list li");
          userItems.forEach((item) => {
            let username = item.getAttribute("data-username");
            let unreadCount = unreadCounts[username] || 0;
            let badge = document.getElementById(
              "notification-badge-" + username
            );
            if (badge) {
              badge.innerText = unreadCount;
              badge.style.display = unreadCount > 0 ? "inline-block" : "none";
            }
          });
          let totalUnread = Object.values(unreadCounts).reduce(
            (sum, count) => sum + count,
            0
          );
          if (totalUnread > 0) {
            $("#chat-notification").show();
          } else {
            $("#chat-notification").hide();
          }
        }
      });