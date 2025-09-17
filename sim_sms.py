# sim_sms.py - SMS via SIM7600G-H module
import serial
import time
from datetime import datetime
import os

class SIM7600SMS:
    def __init__(self, port='/dev/ttyUSB2', baudrate=115200):
        """Initialize SIM7600G-H connection"""
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.connect()
    
    def connect(self):
        """Connect to SIM7600G-H module"""
        try:
            print(f"Connecting to SIM7600G-H on {self.port}...")
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=10,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            time.sleep(2)
            if self.send_at_command("AT"):
                print("✓ SIM7600G-H connected")
                self.setup_sms()
            else:
                raise Exception("AT command failed")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            raise

    def send_at_command(self, command, timeout=5):
        """Send AT command and read response"""
        if not self.ser or not self.ser.is_open:
            return False
        self.ser.reset_input_buffer()
        self.ser.write((command + '\r\n').encode())
        start_time = time.time()
        response = ""
        while time.time() - start_time < timeout:
            if self.ser.in_waiting > 0:
                response += self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                if "OK" in response or "ERROR" in response:
                    break
            time.sleep(0.1)
        print(f"AT: {command} -> {response.strip()}")
        return "OK" in response

    def setup_sms(self):
        """Set SMS to text mode and check network"""
        print("Setting up SMS...")
        if not self.send_at_command("AT+CMGF=1"):
            raise Exception("Failed to set SMS text mode")
        self.send_at_command("AT+CSQ")      # Signal strength
        self.send_at_command("AT+CREG?")    # Network registration
        print("✓ SMS setup complete")

    def send_sms(self, phone_number, message):
        """Send SMS to a Philippines number (+63)"""
        try:
            print(f"Sending SMS to {phone_number}...")
            # Format number for PH
            if not phone_number.startswith('+'):
                if len(phone_number) == 10:
                    phone_number = '+63' + phone_number
                else:
                    raise ValueError("Phone number must be 10 digits for Philippines.")
            
            # Start SMS
            self.ser.reset_input_buffer()
            self.ser.write(f'AT+CMGS="{phone_number}"\r\n'.encode())
            time.sleep(1)
            self.ser.write((message + '\r\n').encode())
            time.sleep(1)
            self.ser.write(bytes([26]))  # Ctrl+Z

            # Wait for response
            start_time = time.time()
            response = ""
            while time.time() - start_time < 30:
                if self.ser.in_waiting > 0:
                    response += self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    if "+CMGS:" in response or "ERROR" in response:
                        break
                time.sleep(0.5)
            
            if "+CMGS:" in response:
                print(f"✓ SMS sent successfully to {phone_number}")
                return True
            else:
                print(f"✗ SMS failed: {response}")
                return False
        except Exception as e:
            print(f"✗ SMS send error: {e}")
            return False

    def check_network(self):
        """Check signal and network registration"""
        print("Checking network...")
        self.send_at_command("AT+CSQ")
        self.send_at_command("AT+CREG?")
        self.send_at_command("AT+COPS?")

    def close(self):
        """Close serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("✓ SIM7600G-H disconnected")

def find_sim7600_port():
    """Automatically find SIM7600G-H port"""
    possible_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3', '/dev/ttyACM0', '/dev/ttyACM1']
    for port in possible_ports:
        if os.path.exists(port):
            try:
                test_ser = serial.Serial(port, 115200, timeout=2)
                test_ser.write(b'AT\r\n')
                time.sleep(1)
                if test_ser.in_waiting > 0:
                    resp = test_ser.read(test_ser.in_waiting).decode('utf-8', errors='ignore')
                    test_ser.close()
                    if "OK" in resp:
                        print(f"✓ Found SIM7600G-H on {port}")
                        return port
            except:
                continue
    print("✗ SIM7600G-H not found")
    return None

def test_sms():
    """Test sending an SMS"""
    port = find_sim7600_port()
    if not port:
        port = '/dev/ttyUSB2'  # Default to USB2 if not found automatically
    sms = SIM7600SMS(port)
    sms.check_network()
    phone = input("Enter PH phone number (10 digits): ")
    message = f"Security test - {datetime.now().strftime('%H:%M:%S')}"
    sms.send_sms(phone, message)
    sms.close()

if __name__ == "__main__":
    test_sms()
