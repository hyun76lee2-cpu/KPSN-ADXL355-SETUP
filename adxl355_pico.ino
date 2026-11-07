/*
  ADXL355 (EVAL-ADXL355-PMDZ) → 라즈베리파이 피코 → USB 시리얼 → imm_live.py
  보드: arduino-pico (Earle Philhower) 코어의 "Raspberry Pi Pico"

  배선 (SPI0, ADI 문서 기준)
    PMOD 1번 CS    → GP17 (22번 핀)
    PMOD 2번 MOSI  → GP19 (25번 핀)
    PMOD 3번 MISO  → GP16 (21번 핀)
    PMOD 4번 SCLK  → GP18 (24번 핀)
    PMOD 5번 GND   → GND  (23번 핀)
    PMOD 6번 VDD   → 3V3(OUT) (36번 핀)
    PMOD 10번 DRDY → GP20 (26번 핀)   이 스케치에서는 쓰지 않음 (연결해 둬도 무방)

  진단 (보드의 초록 LED)
    빠르게 깜빡임(0.1초)  : 시리얼 모니터나 imm_live.py가 포트를 열기를 기다리는 중
    느리게 깜빡임(1초)    : 센서를 찾지 못함. 시리얼에 원인이 나옴
    켜짐/꺼짐이 아주 짧게 반복 : 정상 측정 중 (0.5초마다 토글)

  동작
    시작할 때 ID 레지스터 3개를 읽어 배선을 스스로 점검하고, 결과를 '#' 줄로 알려준다.
    측정 범위 ±2g (1카운트 = 3.9 µg ≈ 0.00383 gal), 1000 Hz로 읽어 10개씩 평균해 100 Hz로 보낸다.
    한 줄 형식: micros,합x,합y,합z,개수   (ADXL345 스케치와 같음. imm_live.py가 그대로 받음)
*/
#include <SPI.h>

const uint8_t PIN_CS = 17;
const uint8_t AVG = 10;                 // 1000 Hz / 10 = 100 Hz
SPISettings spiCfg(8000000, MSBFIRST, SPI_MODE0);

// ADXL355 레지스터
const uint8_t REG_DEVID_AD = 0x00;      // 0xAD
const uint8_t REG_DEVID_MST = 0x01;     // 0x1D
const uint8_t REG_PARTID = 0x02;        // 0xED
const uint8_t REG_STATUS = 0x04;        // bit0 DATA_RDY
const uint8_t REG_XDATA3 = 0x08;        // X3 X2 X1 Y3 Y2 Y1 Z3 Z2 Z1 (9바이트)
const uint8_t REG_FILTER = 0x28;        // 하위 4비트: 출력 속도 (0010 = 1000 Hz, 저역통과 250 Hz)
const uint8_t REG_RANGE = 0x2C;         // 하위 2비트: 01 = ±2g
const uint8_t REG_POWER_CTL = 0x2D;     // 0 = 측정
const uint8_t REG_RESET = 0x2F;         // 0x52를 쓰면 초기화

void writeReg(uint8_t reg, uint8_t val) {
  SPI.beginTransaction(spiCfg);
  digitalWrite(PIN_CS, LOW);
  SPI.transfer((reg << 1) | 0);          // 쓰기: 주소 << 1, 마지막 비트 0
  SPI.transfer(val);
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();
}

void readRegs(uint8_t reg, uint8_t *buf, uint8_t n) {
  SPI.beginTransaction(spiCfg);
  digitalWrite(PIN_CS, LOW);
  SPI.transfer((reg << 1) | 1);          // 읽기: 주소 << 1, 마지막 비트 1
  for (uint8_t i = 0; i < n; i++) buf[i] = SPI.transfer(0x00);
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();
}

uint8_t readReg(uint8_t reg) {
  uint8_t v;
  readRegs(reg, &v, 1);
  return v;
}

int32_t toInt20(const uint8_t *b) {      // 20비트 2의 보수 → 정수
  int32_t v = ((int32_t)b[0] << 12) | ((int32_t)b[1] << 4) | (b[2] >> 4);
  if (v & 0x80000) v -= 0x100000;
  return v;
}

bool checkWiring() {
  uint8_t id[3];
  readRegs(REG_DEVID_AD, id, 3);
  if (id[0] == 0xAD && id[1] == 0x1D && id[2] == 0xED) {
    Serial.println("# ADXL355 확인됨 (ID AD 1D ED). 배선 정상");
    return true;
  }
  Serial.print("# ADXL355를 찾지 못했어요. 읽은 ID: ");
  for (uint8_t i = 0; i < 3; i++) { Serial.print(id[i], HEX); Serial.print(' '); }
  if ((id[0] == 0xFF && id[1] == 0xFF) || (id[0] == 0x00 && id[1] == 0x00)) {
    Serial.println(" → 전원(6번→36번), GND(5번→23번), MISO(3번→GP16) 연결을 확인하세요");
  } else {
    Serial.println(" → CS(1번→GP17), SCLK(4번→GP18), MOSI(2번→GP19)가 서로 바뀌지 않았는지 확인하세요");
  }
  return false;
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);                  // 피코는 USB 직결이라 속도 값과 상관없음
  while (!Serial) {                      // 포트가 열릴 때까지 대기 (첫 메시지를 놓치지 않게)
    digitalWrite(LED_BUILTIN, (millis() / 100) % 2);
    delay(10);
  }
  Serial.println("# 피코 시작 (adxl355_pico 진단판)");
  pinMode(PIN_CS, OUTPUT);
  digitalWrite(PIN_CS, HIGH);
  SPI.begin();                           // arduino-pico 기본 SPI0 핀: GP16 MISO, GP18 SCK, GP19 MOSI
  delay(20);

  writeReg(REG_RESET, 0x52);
  delay(20);
  while (!checkWiring()) {               // 맞게 연결될 때까지 2초마다 다시 점검
    for (int i = 0; i < 4; i++) { digitalWrite(LED_BUILTIN, i % 2 == 0); delay(500); }
  }

  writeReg(REG_RANGE, 0x01);             // ±2g
  writeReg(REG_FILTER, 0x02);            // 1000 Hz, 고역통과 끔
  writeReg(REG_POWER_CTL, 0x00);         // 측정 시작
  delay(10);
  Serial.println("# ADXL355 ±2g 1000Hz avg10 -> 100Hz, columns: micros,sx,sy,sz,n");
}

long sx = 0, sy = 0, sz = 0;
uint8_t n = 0;
unsigned long lastData = 0, lastWarn = 0, lastLed = 0;

void loop() {
  unsigned long now = millis();
  if (now - lastLed > 500) { lastLed = now; digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); }
  uint8_t st = readReg(REG_STATUS);
  if (!(st & 0x01)) {                            // 새 측정값이 아직 없음
    if (now - lastData > 1000 && now - lastWarn > 1000) {
      lastWarn = now;
      Serial.print("# 1초 넘게 새 측정값이 없어요. STATUS=0x");
      Serial.print(st, HEX);
      Serial.print(", POWER_CTL=0x");
      Serial.println(readReg(REG_POWER_CTL), HEX);
    }
    return;
  }
  lastData = now;

  uint8_t b[9];
  readRegs(REG_XDATA3, b, 9);
  sx += toInt20(b);
  sy += toInt20(b + 3);
  sz += toInt20(b + 6);
  n++;

  if (n >= AVG) {
    Serial.print(micros()); Serial.print(',');
    Serial.print(sx); Serial.print(',');
    Serial.print(sy); Serial.print(',');
    Serial.print(sz); Serial.print(',');
    Serial.println(n);
    sx = sy = sz = 0; n = 0;
  }
}
