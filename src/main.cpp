#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);
#include <PubSubClient.h>

const char *ssid = "Araspot";
const char *password = "yondatau";
const char *mqtt_server = "test.mosquitto.org";

WiFiClient espClient;
PubSubClient client(espClient);

long now = millis();
long lastMeasure = 0;
String macAddr = "";

void setup_wifi()
{
    delay(10);
    Serial.println();
    Serial.print("Connecting to ");
    Serial.println(ssid);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.print("WiFi connected - ESP IP address: ");
    Serial.println(WiFi.localIP());
    macAddr = WiFi.macAddress();
    Serial.println(macAddr);
}

void reconnect()
{
    while (!client.connected())
    {
        Serial.print("Attempting MQTT connection...");
        if (client.connect(macAddr.c_str()))
        {
            Serial.println("connected");
        }
        else
        {
            Serial.print("failed, rc=");
            Serial.print(client.state());
            Serial.println(" try again in 5 seconds");
            delay(5000);
        }
    }
}

// void setup()
// {
//   Serial.begin(115200);
//   Serial.println("Mqtt Node-RED");
//   setup_wifi();
//   client.setServer(mqtt_server, 1883);
// }

// void loop()
// {
//   if (!client.connected())
//   {
//     reconnect();
//   }
//   if (!client.loop())
//   {
//     client.connect(macAddr.c_str());
//   }
//   now = millis();
//   if (now - lastMeasure > 5000)
//   {
//     lastMeasure = now;

//     int rssi = WiFi.RSSI();

//     static char RSSITemp[7];
//     dtostrf(rssi, 4, 2, RSSITemp);
//     Serial.println(RSSITemp);

//     client.publish("2041720230/room/rssi", RSSITemp);
//   }
// }

////////////////////////////////////////////////

// void setup()
// {
//     Serial.begin(115200);
//     WiFi.mode(WIFI_STA);
//     WiFi.disconnect();
// }

// void loop()
// {
//     int numNetwork = WiFi.scanNetworks();

//     for (int i = 0; i < numNetwork; i++)
//     {
//         Serial.println(WiFi.SSID(i));

//         delay(3000);
//     }
// }

//////////////////////////////

const char *ssid1 = "Araspot";
const char *ssid2 = "JTI-POLINEMA";
const char *ssid3 = "gudang";
const char *ssid4 = "Auditorium LT8 Barat";
const char *ssid5 = "Wifi Barat";

void setup()
{
    Serial.begin(115200);
    WiFi.mode(WIFI_STA);
    delay(100);

    setup_wifi();
    client.setServer(mqtt_server, 1883);

    lcd.init();
    lcd.backlight();
    lcd.clear();
    lcd.home();

    Serial.println();
    Serial.println("Memulai...");
}

int32_t getRSSI(const char *ssid)
{
    int32_t rssi = 0;

    int32_t numberOfNetworks = WiFi.scanNetworks();

    for (int32_t i = 0; i < numberOfNetworks; i++)
    {
        if (WiFi.SSID(i) == ssid)
        {
            rssi = WiFi.RSSI(i);
            break;
        }
    }

    WiFi.scanDelete();

    return rssi;
}

// void loop()
// {
//     int32_t rssi1 = getRSSI(ssid1);
//     int32_t rssi2 = getRSSI(ssid2);
//     int32_t rssi3 = getRSSI(ssid3);
//     int32_t rssi4 = getRSSI(ssid4);
//     int32_t rssi5 = getRSSI(ssid5);

//     Serial.print("RSSI ");
//     Serial.print(ssid1);
//     Serial.print(": ");
//     Serial.println(rssi1);

//     Serial.print("RSSI ");
//     Serial.print(ssid2);
//     Serial.print(": ");
//     Serial.println(rssi2);

//     Serial.print("RSSI ");
//     Serial.print(ssid3);
//     Serial.print(": ");
//     Serial.println(rssi3);

//     Serial.print("RSSI ");
//     Serial.print(ssid4);
//     Serial.print(": ");
//     Serial.println(rssi4);

//     Serial.print("RSSI ");
//     Serial.print(ssid5);
//     Serial.print(": ");
//     Serial.println(rssi5);

//     Serial.println("----------------");

//     delay(1000);
// }

///////////////////////////////////////////////////////////

#include "model.h"                              //the classifier model
Eloquent::ML::Port::SVM SVM_SKleanr_classifier; // instanciate the classifier object

// // void setup()
// // {
// //     Serial.begin(115200); // begin the Serial communication
// // }

void loop()
{
    float rssi1 = getRSSI(ssid1);
    float rssi2 = getRSSI(ssid2);
    float rssi3 = getRSSI(ssid3);
    float rssi4 = getRSSI(ssid4);
    float rssi5 = getRSSI(ssid5);

    float features[] = {rssi1, rssi2, rssi3, rssi4, rssi5}; // create the array containing the read values size is : 1x3 same as defined in google colab

    String output_str = SVM_SKleanr_classifier.predictLabel(features); // run inference

    if (!client.connected())
    {
        reconnect();
    }
    if (!client.loop())
    {
        client.connect(macAddr.c_str());
    }
    now = millis();
    if (now - lastMeasure > 5000)
    {
        lcd.home();
        lcd.print("Sekarang Anda ");
        lcd.setCursor(0, 1); // set cursor to column 0, row 1
        lcd.print("Ada di : ");
        lcd.print(String(output_str));

        lastMeasure = now;

        // static char RSSITemp[7];
        // // dtostrf(static_cast<double>(output_str.toFloat()), 4, 2, RSSITemp);
        // dtostrf(output_str, 4, 2, RSSITemp);
        Serial.println(output_str);

        client.publish("2041720230/room/ips", output_str.c_str());
    }

    Serial.println(output_str); // print the resulting orientation
    delay(700);
}