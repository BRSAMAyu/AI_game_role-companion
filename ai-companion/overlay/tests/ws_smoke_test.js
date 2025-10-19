/* eslint-disable no-console */
const { WebSocketServer, WebSocket } = require('ws');

function runSmokeTest() {
  return new Promise((resolve, reject) => {
    const server = new WebSocketServer({ port: 0 }, () => {
      const address = server.address();
      const url = `ws://127.0.0.1:${address.port}`;
      const client = new WebSocket(url);

      const payload = { type: 'showText', text: 'hi', ms: 1200, style: 'battle' };

      const timer = setTimeout(() => {
        reject(new Error('WebSocket message not received in time'));
      }, 2000);

      server.on('connection', socket => {
        socket.on('message', data => {
          clearTimeout(timer);
          try {
            const parsed = JSON.parse(data.toString());
            if (parsed.type === payload.type && parsed.text === payload.text && parsed.style === payload.style) {
              resolve();
            } else {
              reject(new Error('Received unexpected payload'));
            }
          } finally {
            socket.close();
            server.close();
          }
        });
      });

      client.on('open', () => {
        client.send(JSON.stringify(payload));
        client.close();
      });

      client.on('error', err => {
        clearTimeout(timer);
        server.close();
        reject(err);
      });
    });

    server.on('error', err => {
      reject(err);
    });
  });
}

runSmokeTest()
  .then(() => {
    console.log('WS smoke test passed');
  })
  .catch(err => {
    console.error('WS smoke test failed:', err.message);
    process.exitCode = 1;
  });
