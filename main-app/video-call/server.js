const express = require('express');
const http = require('http');
const socketio = require('socket.io');

const app = express();
app.use(express.static(__dirname));

// Create HTTP server instead of HTTPS
const server = http.createServer(app);

// Create Socket.io server
const io = socketio(server, {
    cors: {
        origin: [
            "http://localhost:8181",
            // 'http://LOCAL-DEV-IP-HERE' // if using a phone or another computer
        ],
        methods: ["GET", "POST"]
    }
});

// Start server
server.listen(8181, () => {
    console.log("Server running on http://localhost:8181");
});

// Offers and connected sockets
const offers = [];
const connectedSockets = [];

io.on('connection', (socket) => {
    const userName = socket.handshake.auth.userName;
    const password = socket.handshake.auth.password;

    if (password !== "x") {
        socket.disconnect(true);
        return;
    }

    connectedSockets.push({ socketId: socket.id, userName });

    // Send available offers to new client
    if (offers.length) {
        socket.emit('availableOffers', offers);
    }

    socket.on('newOffer', newOffer => {
        offers.push({
            offererUserName: userName,
            offer: newOffer,
            offerIceCandidates: [],
            answererUserName: null,
            answer: null,
            answererIceCandidates: []
        });

        socket.broadcast.emit('newOfferAwaiting', offers.slice(-1));
    });

    socket.on('newAnswer', (offerObj, ackFunction) => {
        const socketToAnswer = connectedSockets.find(s => s.userName === offerObj.offererUserName);
        if (!socketToAnswer) return;

        const offerToUpdate = offers.find(o => o.offererUserName === offerObj.offererUserName);
        if (!offerToUpdate) return;

        ackFunction(offerToUpdate.offerIceCandidates);

        offerToUpdate.answer = offerObj.answer;
        offerToUpdate.answererUserName = userName;

        socket.to(socketToAnswer.socketId).emit('answerResponse', offerToUpdate);
    });

    socket.on('sendIceCandidateToSignalingServer', iceCandidateObj => {
        const { didIOffer, iceUserName, iceCandidate } = iceCandidateObj;

        if (didIOffer) {
            const offerInOffers = offers.find(o => o.offererUserName === iceUserName);
            if (offerInOffers) {
                offerInOffers.offerIceCandidates.push(iceCandidate);

                if (offerInOffers.answererUserName) {
                    const socketToSendTo = connectedSockets.find(s => s.userName === offerInOffers.answererUserName);
                    if (socketToSendTo) {
                        socket.to(socketToSendTo.socketId).emit('receivedIceCandidateFromServer', iceCandidate);
                    } else {
                        console.log("Ice candidate received but could not find answerer");
                    }
                }
            }
        } else {
            const offerInOffers = offers.find(o => o.answererUserName === iceUserName);
            if (!offerInOffers) return;

            const socketToSendTo = connectedSockets.find(s => s.userName === offerInOffers.offererUserName);
            if (socketToSendTo) {
                socket.to(socketToSendTo.socketId).emit('receivedIceCandidateFromServer', iceCandidate);
            } else {
                console.log("Ice candidate received but could not find offerer");
            }
        }
    });
});
