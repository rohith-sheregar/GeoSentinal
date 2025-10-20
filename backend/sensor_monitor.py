import serial
import json
import paho.mqtt.client as mqtt
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from twilio.rest import Client
import os
import numpy as np
from datetime import datetime

class SensorMonitor:
    def __init__(self):
        self.sensor_data = {
            'displacement': [],
            'strain': [],
            'pore_pressure': [],
            'rainfall': [],
            'temperature': [],
            'vibration': []
        }
        
        # Initialize MQTT client for IoT devices
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_message = self.on_mqtt_message
        
        # Email configuration
        self.sendgrid_client = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        
        # SMS configuration
        self.twilio_client = Client(
            os.getenv('TWILIO_ACCOUNT_SID'),
            os.getenv('TWILIO_AUTH_TOKEN')
        )
        
        # Alert thresholds
        self.thresholds = {
            'displacement': 10.0,  # mm
            'strain': 0.002,      # strain ratio
            'pore_pressure': 50,  # kPa
            'rainfall': 100,      # mm/hr
            'vibration': 5.0      # m/s²
        }

    def connect_mqtt(self, broker="localhost"):
        """Connect to MQTT broker for IoT sensor data"""
        self.mqtt_client.connect(broker)
        self.mqtt_client.subscribe("sensors/#")
        self.mqtt_client.loop_start()

    def on_mqtt_message(self, client, userdata, message):
        """Handle incoming MQTT messages from sensors"""
        try:
            data = json.loads(message.payload)
            sensor_type = message.topic.split('/')[-1]
            if sensor_type in self.sensor_data:
                self.sensor_data[sensor_type].append(data['value'])
                self.check_thresholds(sensor_type, data['value'])
        except json.JSONDecodeError:
            print(f"Error decoding MQTT message: {message.payload}")

    def read_serial_sensors(self, port="/dev/ttyUSB0", baudrate=9600):
        """Read data from serial-connected sensors"""
        try:
            with serial.Serial(port, baudrate) as ser:
                data = ser.readline().decode().strip()
                sensor_data = json.loads(data)
                for sensor_type, value in sensor_data.items():
                    if sensor_type in self.sensor_data:
                        self.sensor_data[sensor_type].append(value)
                        self.check_thresholds(sensor_type, value)
        except serial.SerialException as e:
            print(f"Error reading serial data: {e}")

    def check_thresholds(self, sensor_type, value):
        """Check if sensor values exceed thresholds"""
        if sensor_type in self.thresholds:
            if value > self.thresholds[sensor_type]:
                self.send_alert(
                    f"WARNING: {sensor_type} exceeded threshold",
                    f"{sensor_type} value {value} exceeded threshold of {self.thresholds[sensor_type]}"
                )

    def send_alert(self, subject, message):
        """Send email and SMS alerts"""
        # Send email alert
        try:
            email = Mail(
                from_email='alerts@geosentinel.com',
                to_emails=os.getenv('ALERT_EMAIL'),
                subject=subject,
                plain_text_content=message
            )
            self.sendgrid_client.send(email)
        except Exception as e:
            print(f"Error sending email alert: {e}")

        # Send SMS alert
        try:
            self.twilio_client.messages.create(
                body=message,
                from_=os.getenv('TWILIO_PHONE_NUMBER'),
                to=os.getenv('ALERT_PHONE_NUMBER')
            )
        except Exception as e:
            print(f"Error sending SMS alert: {e}")

    def get_current_data(self):
        """Get the most recent sensor readings"""
        current_data = {}
        for sensor_type, values in self.sensor_data.items():
            if values:
                current_data[sensor_type] = values[-1]
            else:
                current_data[sensor_type] = 0.0
        return current_data

    def get_historical_data(self, hours=24):
        """Get historical sensor data for the specified time period"""
        historical_data = {}
        for sensor_type, values in self.sensor_data.items():
            if values:
                historical_data[sensor_type] = values[-hours:]
            else:
                historical_data[sensor_type] = [0.0] * hours
        return historical_data