#include <Bridge.h>

#include "dht.h"

dht DHT;

#define DHT11_PIN A0
const int pinAdc = A1;
void setup()
{
    Serial.begin(115200);
    Bridge.begin();
}

void send_data()
{
    Serial.print(DHT.humidity, 1);
    Serial.print(",\t");
    Serial.print(DHT.temperature, 1);
    Serial.println();

    Bridge.put("h", String(DHT.humidity));
    Bridge.put("t", String(DHT.temperature));
}

void loop()
{

    Serial.print("DHT11: \t");
    int value = analogRead(A2);
    value = map(value, 0, 800, 0, 200);
    long sum = 0;

    for(int i=0;i<32;i++){
        sum += analogRead(pinAdc);
    }
    sum >>= 5;
    switch (DHT.read11(DHT11_PIN))
    {
      case DHTLIB_OK:
          send_data();
          break;
      case DHTLIB_ERROR_CHECKSUM:
          Serial.println("Checksum error");
          break;
      case DHTLIB_ERROR_TIMEOUT:
          Serial.println("Time out error");
          break;
      case DHTLIB_ERROR_CONNECT:
          Serial.println("Connect error");
          break;
      case DHTLIB_ERROR_ACK_L:
          Serial.println("Ack Low error");
          break;
      case DHTLIB_ERROR_ACK_H:
          Serial.println("Ack High error");
          break;
      default:
          Serial.println("Unknown error");
          break;
    }
    Serial.print(" \t");
    Serial.print("Sound: "); 
    Serial.println(sum);
    Serial.print(" \t");

    Serial.print("Light: "); 
    Serial.println(value);
    
    Bridge.put("s", String(sum));
    Bridge.put("l", String(value));
    delay(1000);
}
