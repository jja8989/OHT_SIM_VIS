const fs = require('fs');

const path = 'app/components/OHTVisualization.tsx';
let src = fs.readFileSync(path, 'utf8');

const before = "transports: ['polling', 'websocket'],";
const after = "transports: ['websocket'],";

if (src.includes(before)) {
  src = src.replace(before, after);
  fs.writeFileSync(path, src);
  console.log('Patched OHTVisualization Socket.IO transport to websocket-only.');
} else if (src.includes(after)) {
  console.log('OHTVisualization is already websocket-only.');
} else {
  throw new Error('Expected Socket.IO transport configuration was not found.');
}
