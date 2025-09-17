# sim_sms.py - SMS via SIM7600G-H module
import serial
import time
import re
from datetime import datetime

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
            
            # Wait for module to be ready
            time.sleep(2)
            
            # Test AT command
            if self.send_at_command("AT"):
                print("✓ SIM7600G-H connected")
                self.setup_sms()
            else:
                raise Exception("AT command failed")
                
        except Exception as e:
            print(f"✗ SIM7600G-H connection failed: {e}")
            print("Check connections:")
            print("- USB cable connected")
            print("- SIM card inserted")
            print("- Module powered on")
            raise
    
    def send_at_command(self, command, timeout=5):
        """Send AT command and get response"""
        try:
            if not self.ser or not self.ser.is_open:
                return False
            
            # Clear input buffer
            self.ser.reset_input_buffer()
            
            # Send command
            self.ser.write((command + '\r\n').encode())
            
            # Read response
            start_time = time.time()
            response = ""
            
            while (time.time() - start_time) < timeout:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    response += data
                    
                    if "OK" in response or "ERROR" in response:
                        break
                time.sleep(0.1)
            
            print(f"AT: {command} -> {response.strip()}")
            return "OK" in response
            
        except Exception as e:
            print(f"AT command error: {e}")
            return False
    
    def setup_sms(self):
        """Setup SMS mode"""
        print("Setting up SMS...")
        
        # Set text mode
        if not self.send_at_command("AT+CMGF=1"):
            raise Exception("Failed to set SMS text mode")
        
        # Check signal strength
        self.send_at_command("AT+CSQ")
        
        # Check network registration
        self.send_at_command("AT+CREG?")
        
        print("✓ SMS setup complete")
    
    def send_sms(self, phone_number, message):
        """Send SMS message"""
        try:
            print(f"Sending SMS to {phone_number}...")
            
            # Ensure phone number format
            if not phone_number.startswith('+'):
                if phone_number.startswith('1') and len(phone_number) == 11:
                    phone_number = '+' + phone_number
                elif len(phone_number) == 10:
                    phone_number = '+1' + phone_number
            
            # Start SMS command
            sms_cmd = f'AT+CMGS="{phone_number}"'
            
            # Clear buffers
            self.ser.reset_input_buffer()
            
            # Send SMS command
            self.ser.write((sms_cmd + '\r\n').encode())
            time.sleep(1)
            
            # Send message
            self.ser.write((message + '\r\n').encode())
            time.sleep(1)
            
            # Send Ctrl+Z to end message
            self.ser.write(bytes([26]))
            
            # Wait for response
            start_time = time.time()
            response = ""
            
            while (time.time() - start_time) < 30:  # SMS can take up to 30 seconds
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    response += data
                    
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
        """Check network status"""
        print("Checking network status...")
        
        # Signal quality
        if self.send_at_command("AT+CSQ"):
            print("Signal check sent")
        
        # Network registration
        if self.send_at_command("AT+CREG?"):
            print("Network registration checked")
        
        # Operator info
        if self.send_at_command("AT+COPS?"):
            print("Operator info retrieved")
    
    def close(self):
        """Close connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("✓ SIM7600G-H disconnected")

def find_sim7600_port():
    """Find SIM7600G-H port automatically"""
    import glob
    
    # Common ports for SIM7600G-H
    possible_ports = [
        '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3',
        '/dev/ttyACM0', '/dev/ttyACM1'
    ]
    
    print("Searching for SIM7600G-H...")
    
    for port in possible_ports:
        if os.path.exists(port):
            try:
                # Test connection
                test_ser = serial.Serial(port, 115200, timeout=2)
                test_ser.write(b'AT\r\n')
                time.sleep(1)
                
                if test_ser.in_waiting > 0:
                    response = test_ser.read(test_ser.in_waiting).decode('utf-8', errors='ignore')
                    test_ser.close()
                    
                    if "OK" in response:
                        print(f"✓ Found SIM7600G-H on {port}")
                        return port
                else:
                    test_ser.close()
                    
            except Exception as e:
                print(f"Port {port}: {e}")
                continue
    
    print("✗ SIM7600G-H not found")
    return None

def test_sms():
    """Test SMS functionality"""
    try:
        # Find port automatically
        port = find_sim7600_port()
        if not port:
            port = input("Enter SIM7600G-H port (e.g., /dev/ttyUSB2): ")
        
        # Initialize SMS
        sms = SIM7600SMS(port)
        
        # Check network
        sms.check_network()
        
        # Test message
        phone = input("Enter phone number to test (+1xxxxxxxxxx): ")
        test_msg = f"Security system test - {datetime.now().strftime('%H:%M:%S')}"
        
        if sms.send_sms(phone, test_msg):
            print("✅ SMS test successful!")
        else:
            print("❌ SMS test failed")
        
        sms.close()
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    import os
    test_sms()
