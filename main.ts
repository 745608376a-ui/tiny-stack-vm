// A complete interpreter for make.fun's tiny stack ISA (kit: generator/subject/ISA.md), matching the
// reference VM (generator/subject/vm.mjs) rule for rule: wrapping 32-bit arithmetic, exact signed and
// unsigned division/remainder, 5-bit shift counts, a 256-value stack, a 64-deep call stack, 256 words of
// memory, all ten traps, and the step and output limits. Per instance it writes stdout.bin (the bytes
// written, kept even on a trap) and exit.txt (0, EXIT's low byte, or the trap's code).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const STACK_MAX = 256, CALLS_MAX = 64, MEM_WORDS = 256, STEP_LIMIT = 1000000, OUT_MAX = 65536;

// Per opcode byte, one packed number: byte*1000 + immediate bytes*100 + popped*10 + pushed. An absent
// byte is an unknown opcode (trap 5). Stack needs are checked first: underflow (trap 2), then the size
// left behind (trap 3), in the reference's order.
const OPS = [
  0, 1101, 2401, 3010, 4012, 5022, 6023, 7033, 16021, 17021, 18021, 19021, 20021, 21021, 22021, 23011,
  24021, 25021, 26021, 27011, 28021, 29021, 30021, 32021, 33021, 34021, 48200, 49210, 50210, 51200,
  52000, 64011, 65020, 80010, 81010, 82001, 83010,
];
const IMM = new Int8Array(256).fill(-1), POPS = new Int8Array(256), PSHS = new Int8Array(256);
for (const v of OPS) {
  const b = (v / 1000) | 0;
  IMM[b] = ((v / 100) | 0) % 10;
  POPS[b] = ((v / 10) | 0) % 10;
  PSHS[b] = v % 10;
}

// Runs one program to its end; returns [stdout bytes, exit code].
function run(prog, inp) {
  const n = prog.length;
  const D = new Int32Array(STACK_MAX);
  const C = new Int32Array(CALLS_MAX);
  const M = new Int32Array(MEM_WORDS);
  const out = new Uint8Array(OUT_MAX);
  let sp = 0, csp = 0, olen = 0, inPos = 0, pc = 0, steps = 0;
  const done = (code) => [out.subarray(0, olen), code];
  for (;;) {
    if (steps === STEP_LIMIT) return done(10);
    if (pc < 0 || pc >= n) return done(9);
    const op = prog[pc];
    const imm = IMM[op];
    if (imm < 0) return done(5);
    if (pc + imm >= n) return done(9); // an immediate cut off by the program's end
    steps++;
    if (sp < POPS[op]) return done(2);
    if (sp - POPS[op] + PSHS[op] > STACK_MAX) return done(3);
    const next = pc + 1 + imm;
    pc = next;
    switch (op) {
      case 0x00: return done(0); // HALT
      case 0x01: D[sp++] = (prog[next - 1] << 24) >> 24; break; // PUSH8: sign-extended
      case 0x02: // PUSH32: little-endian
        D[sp++] = prog[next - 4] | (prog[next - 3] << 8) | (prog[next - 2] << 16) | (prog[next - 1] << 24);
        break;
      case 0x03: sp--; break; // POP
      case 0x04: D[sp] = D[sp - 1]; sp++; break; // DUP
      case 0x05: { const t = D[sp - 1]; D[sp - 1] = D[sp - 2]; D[sp - 2] = t; break; } // SWAP
      case 0x06: D[sp] = D[sp - 2]; sp++; break; // OVER
      case 0x07: { // ROT: ( a b c -- b c a )
        const a = D[sp - 3];
        D[sp - 3] = D[sp - 2]; D[sp - 2] = D[sp - 1]; D[sp - 1] = a;
        break;
      }
      case 0x17: D[sp - 1] = -D[sp - 1] | 0; break; // NEG: wraps
      case 0x1b: D[sp - 1] = ~D[sp - 1]; break; // NOT
      case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16:
      case 0x18: case 0x19: case 0x1a: case 0x1c: case 0x1d: case 0x1e:
      case 0x20: case 0x21: case 0x22: { // binary ops: ( a b -- r ), b is the top
        const b = D[sp - 1], a = D[sp - 2];
        let r;
        if (op === 0x10) r = (a + b) | 0; // ADD
        else if (op === 0x11) r = (a - b) | 0; // SUB
        else if (op === 0x12) r = Math.imul(a, b); // MUL: low 32 bits of the exact product
        else if (op === 0x13) { // DIV: truncated toward zero; INT_MIN / -1 is INT_MIN
          if (b === 0) return done(4);
          r = a === -2147483648 && b === -1 ? a : (a / b) | 0;
        } else if (op === 0x14) { // MOD: sign of a; x % -1 is 0
          if (b === 0) return done(4);
          r = b === -1 ? 0 : a % b | 0;
        } else if (op === 0x15) { // DIVU: operands read as unsigned
          if (b === 0) return done(4);
          r = Math.floor((a >>> 0) / (b >>> 0)) | 0;
        } else if (op === 0x16) { // MODU
          if (b === 0) return done(4);
          r = (a >>> 0) % (b >>> 0) | 0;
        }
        else if (op === 0x18) r = a & b; // AND
        else if (op === 0x19) r = a | b; // OR
        else if (op === 0x1a) r = a ^ b; // XOR
        else if (op === 0x1c) r = a << (b & 31); // SHL: only the low 5 bits of the count
        else if (op === 0x1d) r = a >> (b & 31); // SHR: arithmetic
        else if (op === 0x1e) r = (a >>> (b & 31)) | 0; // SHRU: logical, read back as signed
        else if (op === 0x20) r = a === b ? 1 : 0; // EQ
        else if (op === 0x21) r = a < b ? 1 : 0; // LT: signed
        else r = (a >>> 0) < (b >>> 0) ? 1 : 0; // LTU: unsigned
        D[sp - 2] = r;
        sp--;
        break;
      }
      case 0x30: case 0x31: case 0x32: { // JMP / JZ / JNZ: target = next address + signed offset
        const rel = ((prog[next - 2] | (prog[next - 1] << 8)) << 16) >> 16;
        let take = true;
        if (op !== 0x30) { const v = D[--sp]; take = op === 0x31 ? v === 0 : v !== 0; }
        if (take) pc = next + rel;
        break;
      }
      case 0x33: // CALL: the immediate is an unsigned absolute address
        if (csp === CALLS_MAX) return done(7);
        C[csp++] = next;
        pc = prog[next - 2] | (prog[next - 1] << 8);
        break;
      case 0x34: // RET
        if (csp === 0) return done(8);
        pc = C[--csp];
        break;
      case 0x40: { // LOAD
        const addr = D[sp - 1];
        if (addr < 0 || addr >= MEM_WORDS) return done(6);
        D[sp - 1] = M[addr];
        break;
      }
      case 0x41: { // STORE: ( v addr -- )
        const addr = D[sp - 1], v = D[sp - 2];
        if (addr < 0 || addr >= MEM_WORDS) return done(6);
        sp -= 2;
        M[addr] = v;
        break;
      }
      case 0x50: { // PRINT: decimal, no padding/newline; writes nothing if it would overflow
        const s = String(D[sp - 1]);
        if (olen + s.length > OUT_MAX) return done(11);
        sp--;
        for (let i = 0; i < s.length; i++) out[olen++] = s.charCodeAt(i);
        break;
      }
      case 0x51: // PUTC: one byte, v AND 255
        if (olen + 1 > OUT_MAX) return done(11);
        out[olen++] = D[--sp] & 255;
        break;
      case 0x52: // IN: the next input byte, -1 once the input is used up
        D[sp++] = inPos < inp.length ? inp[inPos++] : -1;
        break;
      case 0x53: return done(D[--sp] & 255); // EXIT: the code is v AND 255
    }
  }
}

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of readdirSync(I).sort()) {
  const prog = readFileSync(join(I, id, 'program.bin'));
  let inp;
  try { inp = readFileSync(join(I, id, 'input.bin')); } catch { inp = Buffer.alloc(0); }
  const [stdout, code] = run(prog, inp);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'stdout.bin'), stdout);
  writeFileSync(join(O, id, 'exit.txt'), code + '\n');
}
