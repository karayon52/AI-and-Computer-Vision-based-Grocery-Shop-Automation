#include <ESP32Servo.h>

// ======================================================
// PIN SETUP
// ======================================================
#define DIR_PIN    12
#define STEP_PIN   14

#define SERVO1_PIN 32
#define SERVO2_PIN 33

// ======================================================
// SERVO OBJECTS
// ======================================================
Servo servo1;
Servo servo2;

// ======================================================
// SERVO ANGLES
// ======================================================
const int SERVO1_HOME = 125;
const int SERVO1_PREP = 30;

const int SERVO2_HOME = 0;
const int SERVO2_PREP = 90;

int servo1Angle = SERVO1_HOME;
int servo2Angle = SERVO2_HOME;

// ======================================================
// STEPPER DISTANCES
// ======================================================
const long CAM_TO_SERVO1_STEPS = 500;
const long CAM_TO_SERVO2_STEPS = 1000;
const long CAM_TO_BASKET3_STEPS = 1500;

// ======================================================
// STEPPER SPEED SETTINGS
// Keep close to your old working style
// ======================================================
int pulseDelayUs = 900;         // your old working speed
int startPulseDelayUs = 1800;   // slower startup for more torque
int endPulseDelayUs   = 1400;   // slower finishing
long rampStepCount    = 120;    // number of ramp steps

// ======================================================
// TIMING
// ======================================================
const int SERVO_MOVE_DELAY_MS = 10;
const int SERVO_SETTLE_MS = 250;
const int ITEM_SETTLE_MS = 250;

bool busy = false;

// ======================================================
// BASIC ONE STEP
// This keeps same style as your old working code
// ======================================================
void oneStepWithCustomDelay(bool dirForward, int customDelayUs) {
  digitalWrite(DIR_PIN, dirForward ? HIGH : LOW);

  digitalWrite(STEP_PIN, HIGH);
  delayMicroseconds(customDelayUs);
  digitalWrite(STEP_PIN, LOW);
  delayMicroseconds(customDelayUs);
}

void oneStep(bool dirForward) {
  oneStepWithCustomDelay(dirForward, pulseDelayUs);
}

// ======================================================
// OLD STYLE FIXED MOVE
// ======================================================
void moveSteps(long steps, bool dirForward) {
  for (long i = 0; i < steps; i++) {
    oneStep(dirForward);
  }
}

// ======================================================
// NEW SAFE RAMP MOVE
// Built on top of your previous working pulse logic
// ======================================================
void moveStepsRamp(long steps, bool dirForward) {
  if (steps <= 0) return;

  long localRamp = rampStepCount;

  if (steps < 2 * localRamp) {
    localRamp = steps / 2;
  }

  if (localRamp < 1) {
    moveSteps(steps, dirForward);
    return;
  }

  for (long i = 0; i < steps; i++) {
    int currentDelay = pulseDelayUs;

    // acceleration region
    if (i < localRamp) {
      currentDelay = startPulseDelayUs - ((startPulseDelayUs - pulseDelayUs) * i) / localRamp;
    }
    // deceleration region
    else if (i >= (steps - localRamp)) {
      long j = i - (steps - localRamp);
      currentDelay = pulseDelayUs + ((endPulseDelayUs - pulseDelayUs) * j) / localRamp;
    }
    // middle region
    else {
      currentDelay = pulseDelayUs;
    }

    oneStepWithCustomDelay(dirForward, currentDelay);
  }
}

// ======================================================
// SERVO HELPERS
// ======================================================
void moveServoSmooth(Servo &sv, int &currentAngle, int targetAngle, int delayMs = SERVO_MOVE_DELAY_MS) {
  targetAngle = constrain(targetAngle, 0, 180);

  if (currentAngle < targetAngle) {
    for (int a = currentAngle; a <= targetAngle; a++) {
      sv.write(a);
      delay(delayMs);
    }
  } else {
    for (int a = currentAngle; a >= targetAngle; a--) {
      sv.write(a);
      delay(delayMs);
    }
  }

  currentAngle = targetAngle;
}

void moveAllHome() {
  moveServoSmooth(servo1, servo1Angle, SERVO1_HOME);
  moveServoSmooth(servo2, servo2Angle, SERVO2_HOME);
}

// ======================================================
// ROUTING LOGIC
// ======================================================
void routeCustomer1() {
  Serial.println("Routing to Customer 1...");

  moveServoSmooth(servo1, servo1Angle, SERVO1_PREP);
  delay(SERVO_SETTLE_MS);

  moveStepsRamp(CAM_TO_SERVO1_STEPS, true);
  delay(ITEM_SETTLE_MS);

  moveServoSmooth(servo1, servo1Angle, SERVO1_HOME);
  delay(400);
}

void routeCustomer2() {
  Serial.println("Routing to Customer 2...");

  moveServoSmooth(servo2, servo2Angle, SERVO2_PREP);
  delay(SERVO_SETTLE_MS);

  moveStepsRamp(CAM_TO_SERVO2_STEPS, true);
  delay(ITEM_SETTLE_MS);

  moveServoSmooth(servo2, servo2Angle, SERVO2_HOME);
  delay(400);
}

void routeCustomer3() {
  Serial.println("Routing to Customer 3...");

  moveStepsRamp(CAM_TO_BASKET3_STEPS, true);
  delay(400);
}

// ======================================================
// COMMAND PARSING
// ======================================================
int parseRouteId(String cmd) {
  int p1 = cmd.indexOf(':');
  int p2 = cmd.indexOf(':', p1 + 1);

  if (p1 < 0 || p2 < 0) return -1;

  String routeText = cmd.substring(p1 + 1, p2);
  return routeText.toInt();
}

String parseLabel(String cmd) {
  int p2 = cmd.indexOf(':', cmd.indexOf(':') + 1);
  if (p2 < 0) return "";
  return cmd.substring(p2 + 1);
}

// ======================================================
// EXECUTE ROUTE
// ======================================================
void executeRoute(int routeId, String label) {
  busy = true;

  Serial.print("ACK:");
  Serial.print(routeId);
  Serial.print(":");
  Serial.println(label);

  moveAllHome();
  delay(200);

  if (routeId == 1) {
    routeCustomer1();
  }
  else if (routeId == 2) {
    routeCustomer2();
  }
  else if (routeId == 3) {
    routeCustomer3();
  }
  else {
    Serial.println("ERR:BAD_ROUTE");
    busy = false;
    return;
  }

  moveAllHome();
  delay(200);

  busy = false;
  Serial.println("DONE");
}

// ======================================================
// MANUAL TEST COMMANDS
// ======================================================
void printMenu() {
  Serial.println("======================================");
  Serial.println("ESP32 Grocery Router");
  Serial.println("Auto commands:");
  Serial.println("ROUTE:1:soap");
  Serial.println("ROUTE:2:cola");
  Serial.println("ROUTE:3:toothpaste");
  Serial.println();
  Serial.println("Manual commands:");
  Serial.println("t1, t2, t3");
  Serial.println("home");
  Serial.println("s1p, s1h, s2p, s2h");
  Serial.println("status");
  Serial.println("vslow, slow, medium, fast");
  Serial.println("rawtest");
  Serial.println("======================================");
  Serial.println();
}

void printStatus() {
  Serial.print("busy = ");
  Serial.println(busy ? "true" : "false");

  Serial.print("servo1Angle = ");
  Serial.println(servo1Angle);

  Serial.print("servo2Angle = ");
  Serial.println(servo2Angle);

  Serial.print("pulseDelayUs = ");
  Serial.println(pulseDelayUs);

  Serial.print("startPulseDelayUs = ");
  Serial.println(startPulseDelayUs);

  Serial.print("endPulseDelayUs = ");
  Serial.println(endPulseDelayUs);

  Serial.print("rampStepCount = ");
  Serial.println(rampStepCount);

  Serial.println();
}

void handleManualCommand(String cmd) {
  if (cmd == "t1") {
    if (busy) { Serial.println("BUSY"); return; }
    executeRoute(1, "manual");
  }
  else if (cmd == "t2") {
    if (busy) { Serial.println("BUSY"); return; }
    executeRoute(2, "manual");
  }
  else if (cmd == "t3") {
    if (busy) { Serial.println("BUSY"); return; }
    executeRoute(3, "manual");
  }
  else if (cmd == "home") {
    moveAllHome();
    Serial.println("Both servos moved to home.");
  }
  else if (cmd == "s1p") {
    moveServoSmooth(servo1, servo1Angle, SERVO1_PREP);
    Serial.println("Servo1 moved to prep.");
  }
  else if (cmd == "s1h") {
    moveServoSmooth(servo1, servo1Angle, SERVO1_HOME);
    Serial.println("Servo1 moved to home.");
  }
  else if (cmd == "s2p") {
    moveServoSmooth(servo2, servo2Angle, SERVO2_PREP);
    Serial.println("Servo2 moved to prep.");
  }
  else if (cmd == "s2h") {
    moveServoSmooth(servo2, servo2Angle, SERVO2_HOME);
    Serial.println("Servo2 moved to home.");
  }
  else if (cmd == "status") {
    printStatus();
  }
  else if (cmd == "rawtest") {
    Serial.println("Running raw stepper test...");
    moveSteps(300, true);
    delay(500);
    moveSteps(300, false);
    Serial.println("Raw test done.");
  }
  else if (cmd == "vslow") {
    pulseDelayUs = 1600;
    startPulseDelayUs = 2600;
    endPulseDelayUs = 2000;
    rampStepCount = 160;
    Serial.println("Stepper speed set to very slow.");
  }
  else if (cmd == "slow") {
    pulseDelayUs = 1200;
    startPulseDelayUs = 2200;
    endPulseDelayUs = 1700;
    rampStepCount = 140;
    Serial.println("Stepper speed set to slow.");
  }
  else if (cmd == "medium") {
    pulseDelayUs = 900;
    startPulseDelayUs = 1800;
    endPulseDelayUs = 1400;
    rampStepCount = 120;
    Serial.println("Stepper speed set to medium.");
  }
  else if (cmd == "fast") {
    pulseDelayUs = 700;
    startPulseDelayUs = 1400;
    endPulseDelayUs = 1100;
    rampStepCount = 100;
    Serial.println("Stepper speed set to fast.");
  }
  else {
    Serial.println("ERR:UNKNOWN_CMD");
    printMenu();
  }
}

// ======================================================
// SETUP
// ======================================================
void setup() {
  Serial.begin(115200);

  pinMode(DIR_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);

  digitalWrite(DIR_PIN, LOW);
  digitalWrite(STEP_PIN, LOW);

  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);

  servo1.attach(SERVO1_PIN, 500, 2400);
  servo2.attach(SERVO2_PIN, 500, 2400);

  servo1.write(SERVO1_HOME);
  servo2.write(SERVO2_HOME);

  servo1Angle = SERVO1_HOME;
  servo2Angle = SERVO2_HOME;

  delay(1000);

  Serial.println("READY");
  printMenu();
  printStatus();
}

// ======================================================
// LOOP
// ======================================================
void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.length() == 0) return;

    if (cmd.startsWith("ROUTE:")) {
      if (busy) {
        Serial.println("BUSY");
        return;
      }

      int routeId = parseRouteId(cmd);
      String label = parseLabel(cmd);

      executeRoute(routeId, label);
    } else {
      handleManualCommand(cmd);
    }
  }
}