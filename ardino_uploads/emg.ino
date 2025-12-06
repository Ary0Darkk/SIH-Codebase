/*
  Arduino sketch: ICM-20948 (SparkFun lib) + EMG envelope -> Serial
  Sends CSV: ax,ay,az,emg
  - Requires SparkFun ICM-20948 Arduino Library
  - EMG connected to A0
  - ICM-20948 connected via I2C (SDA / SCL)
*/

#include <Wire.h>
#include "SparkFun_ICM-20948_ArduinoLibrary.h" // install via Library Manager
ICM_20948_I2C myICM;

const int EMG_PIN = A0;
const float EMG_VREF = 5.0;         // change to 3.3 if using 3.3V board
const int ADC_MAX = 1023;           // 10-bit ADC default on UNO; change for 12-bit boards
const float SAMPLE_RATE_HZ = 200.0; // target loop rate (approx)

// EMG envelope filter params (exponential smoothing)
const float EMG_ALPHA = 0.05; // smoothing factor (0..1) - lower is smoother
float emg_envelope = 0;

unsigned long lastMicros = 0;
unsigned long intervalMicros = (unsigned long)(1000000.0 / SAMPLE_RATE_HZ);

void setup()
{
    Serial.begin(115200);
    delay(200);

    Wire.begin();
    if (myICM.begin() != ICM_20948_Stat_Ok)
    {
        Serial.println("ICM-20948 initialization failed. Check wiring / library.");
        while (1)
        {
            delay(1000);
        }
    }

    // Configure accel full-scale to a reasonable range if desired (example: +-16g)
    // NOTE: the SparkFun library may have different API names for configuring ranges.
    // Consult the library docs if you want to change accel/gyro scales.

    // Warm-up delay
    delay(100);
    lastMicros = micros();
    Serial.println("ARDUINO IMU+EMG READY");
}

void loop()
{
    // keep approximate fixed sample rate
    unsigned long now = micros();
    if (now - lastMicros < intervalMicros)
        return;
    lastMicros = now;

    // --- Read IMU ---
    // Ask the library to update internal data
    myICM.getAGMT(); // get accel/gyro/mag/temp; (method name from SparkFun lib)
    // The SparkFun library stores results in myICM._agmt (struct) (if using different lib adapt)
    // NOTE: fields are typically INT16 raw values; check library for conversion functions.
    // We'll extract raw accel values (16-bit) and send them as integers.
    int16_t ax_raw = myICM._agmt.acc.x; // raw accelerometer X
    int16_t ay_raw = myICM._agmt.acc.y;
    int16_t az_raw = myICM._agmt.acc.z;

    // Optionally convert to milli-g (mg) or m/s^2 if you want:
    // Example (if accel range is +-16g and raw is 16-bit signed; adjust per config):
    // float ax_mg = (float)ax_raw * (scale_factor);
    // For simplicity we'll send raw integer values; Python code interprets them.

    // --- Read EMG and compute envelope ---
    int raw = analogRead(EMG_PIN); // 0..ADC_MAX
    // Convert ADC to signed centered value assuming sensor resting at mid-rail:
    float centered = (float)raw - (ADC_MAX / 2.0); // EMG modules often center at mid-rail
    // Rectify (absolute)
    float rectified = fabs(centered);

    // Low-pass (exponential) to make envelope
    emg_envelope = (EMG_ALPHA * rectified) + ((1.0 - EMG_ALPHA) * emg_envelope);

    // Optionally scale envelope into 0..1000 range for easier thresholds:
    float emg_out = emg_envelope; // leave as-is; Python code will calibrate threshold

    // --- Print CSV: ax,ay,az,emg  ---
    Serial.print(ax_raw);
    Serial.print(",");
    Serial.print(ay_raw);
    Serial.print(",");
    Serial.print(az_raw);
    Serial.print(",");
    Serial.println((int)emg_out); // send envelope as int for simpler parsing
}