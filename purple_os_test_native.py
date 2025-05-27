#!/usr/bin/env python3
"""
Purple OS Automated Testing Framework - Native SSH Version
=========================================================
Enhanced testing framework using native SSH (subprocess) instead of paramiko
Includes parallel execution, comprehensive reporting, and dashboard generation

Author: Purple OS Test Automation Team
Version: 2.0.0
"""

import os
import sys
import json
import time
import logging
import datetime
import subprocess
import concurrent.futures
import threading
import queue
import shutil
import re
import yaml
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from jinja2 import Template
import tempfile
import hashlib

# Configuration constants
DEVICE_IP = "192.168.1.1"
DEVICE_USERNAME = "root"
SSH_PORT = 22
MAX_WORKERS = 10
TIMEOUT = 30
LOG_DIR = Path("./test_logs")
REPORT_DIR = Path("./test_reports")
DASHBOARD_DIR = Path("./test_dashboards")

# Determine local folder based on OS
if sys.platform == "win32":
    LOCAL_FOLDER = Path(os.environ.get('USERPROFILE', '.')) / "Desktop"
else:
    LOCAL_FOLDER = Path.home() / "Desktop"


@dataclass
class TestResult:
    """Data class for storing individual test results"""
    test_name: str
    test_category: str
    status: str  # PASS, FAIL, ERROR, SKIP
    start_time: datetime.datetime
    end_time: datetime.datetime
    duration: float
    output: str
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceInfo:
    """Data class for storing device information"""
    model: str = ""
    firmware_version: str = ""
    kernel_version: str = ""
    uptime: str = ""
    memory_total: str = ""
    memory_free: str = ""
    cpu_info: str = ""
    network_interfaces: List[Dict[str, str]] = field(default_factory=list)
    variant: str = ""


class NativeSSHConnection:
    """SSH connection using subprocess and native SSH client"""
    
    def __init__(self, host="192.168.1.1", username="root", port=22):
        self.host = host
        self.username = username
        self.port = port
        self.ssh_base = [
            "ssh", 
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "PreferredAuthentications=publickey,password",
            "-o", "PasswordAuthentication=yes",
            "-p", str(port),
            f"{username}@{host}"
        ]
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute_command(self, command, timeout=30):
        """Execute command via SSH"""
        full_command = self.ssh_base + [command]
        
        try:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='ignore'
            )
            
            return result.stdout, result.stderr, result.returncode
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out: {command}")
            return "", "Command timed out", 1
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return "", str(e), 1
    
    def test_connection(self):
        """Test if SSH connection works"""
        stdout, stderr, code = self.execute_command("echo 'Connection OK'", timeout=5)
        return code == 0 and "Connection OK" in stdout
    
    def transfer_file(self, local_path, remote_path):
        """Transfer file using SCP"""
        scp_command = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-P", str(self.port),
            local_path,
            f"{self.username}@{self.host}:{remote_path}"
        ]
        
        try:
            result = subprocess.run(scp_command, capture_output=True, text=True, timeout=60)
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"SCP transfer failed: {e}")
            return False
    
    def download_file(self, remote_path, local_path):
        """Download file using SCP"""
        scp_command = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-P", str(self.port),
            f"{self.username}@{self.host}:{remote_path}",
            local_path
        ]
        
        try:
            result = subprocess.run(scp_command, capture_output=True, text=True, timeout=60)
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"SCP download failed: {e}")
            return False
    
    def download_directory(self, remote_path, local_path):
        """Download directory recursively using SCP"""
        scp_command = [
            "scp",
            "-r",  # Recursive
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-P", str(self.port),
            f"{self.username}@{self.host}:{remote_path}",
            local_path
        ]
        
        try:
            result = subprocess.run(scp_command, capture_output=True, text=True, timeout=120)
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"SCP directory download failed: {e}")
            return False


class SSHConnectionPool:
    """Pool of Native SSH connections for parallel execution"""
    
    def __init__(self, host, username, port=22, max_connections=10):
        self.host = host
        self.username = username
        self.port = port
        self.max_connections = max_connections
        self._pool = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize connection pool
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool with SSH connections"""
        for i in range(self.max_connections):
            conn = NativeSSHConnection(self.host, self.username, self.port)
            if conn.test_connection():
                self._pool.put(conn)
                self.logger.info(f"Created connection {i+1}/{self.max_connections}")
            else:
                self.logger.error(f"Failed to create connection {i+1}")
    
    def get_connection(self) -> NativeSSHConnection:
        """Get a connection from the pool"""
        try:
            return self._pool.get(timeout=30)
        except queue.Empty:
            self.logger.warning("No connections available, creating new one")
            return NativeSSHConnection(self.host, self.username, self.port)
    
    def return_connection(self, conn: NativeSSHConnection):
        """Return a connection to the pool"""
        try:
            self._pool.put(conn, block=False)
        except queue.Full:
            self.logger.warning("Connection pool full")


class PurpleOSTestFramework:
    """Main testing framework for Purple OS devices using native SSH"""
    
    def __init__(self, device_ip=DEVICE_IP, username=DEVICE_USERNAME, max_workers=MAX_WORKERS):
        self.device_ip = device_ip
        self.username = username
        self.max_workers = max_workers
        self.test_results: List[TestResult] = []
        self.device_info = DeviceInfo()
        
        # Create directories
        self._create_directories()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize connection pool
        self.connection_pool = SSHConnectionPool(device_ip, username, max_connections=max_workers)
        
        # Test connection
        conn = self.connection_pool.get_connection()
        if not conn.test_connection():
            raise Exception(f"Cannot connect to device at {device_ip}")
        self.connection_pool.return_connection(conn)
        
        # Device-specific attributes
        self.variant_connected = ""
        self.local_path = None
        self.local_folder = self.config.get('local_folder', LOCAL_FOLDER)
    
    def _create_directories(self):
        """Create necessary directories"""
        for directory in [LOG_DIR, REPORT_DIR, DASHBOARD_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> dict:
        """Load configuration from file or use defaults"""
        config_file = Path("config.yaml")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                self.logger.warning(f"Could not load config: {e}")
        
        # Return default config
        return {
            'local_folder': str(LOCAL_FOLDER),
            'tr181_parameters': {
                'device_info': [
                    "Device.DeviceInfo.Manufacturer",
                    "Device.DeviceInfo.ModelName",
                    "Device.DeviceInfo.SerialNumber",
                    "Device.DeviceInfo.SoftwareVersion",
                    "Device.DeviceInfo.HardwareVersion"
                ],
                'network': [
                    "Device.LAN.IPAddress",
                    "Device.LAN.SubnetMask",
                    "Device.WiFi.Radio.1.Enable"
                ]
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_file = LOG_DIR / f"purple_os_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(self.__class__.__name__)
    
    def execute_command(self, command: str, timeout: int = TIMEOUT) -> Tuple[str, str, int]:
        """Execute command on device using connection pool"""
        conn = self.connection_pool.get_connection()
        try:
            stdout, stderr, code = conn.execute_command(command, timeout)
            return stdout, stderr, code
        finally:
            self.connection_pool.return_connection(conn)
    
    def get_device_variant(self) -> str:
        """Get device variant/platform information"""
        self.logger.info("Getting device variant...")
        
        stdout, stderr, code = self.execute_command("uci show version")
        
        if code != 0:
            self.logger.error(f"Failed to get device variant: {stderr}")
            # Try alternative
            stdout, stderr, code = self.execute_command("cat /etc/device_info")
        
        desired_key = "version.@version[0].product="
        variant_connected = ""
        
        for line in stdout.split('\n'):
            if desired_key in line:
                value = line.split("=")[1].strip("'\"")
                variant_connected = value
                self.logger.info(f"Platform connected is {variant_connected}")
                break
        
        if not variant_connected:
            variant_connected = "Unknown_Device"
            self.logger.warning("Could not determine device variant, using 'Unknown_Device'")
        
        self.variant_connected = variant_connected
        self.device_info.variant = variant_connected
        
        # Create local folder for results
        self.local_path = Path(self.local_folder) / f"{variant_connected}_results_folder"
        self.local_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created local folder: {self.local_path}")
        
        return variant_connected
    
    def collect_device_info(self) -> DeviceInfo:
        """Collect comprehensive device information"""
        self.logger.info("Collecting device information...")
        
        # Get variant first
        if not self.variant_connected:
            self.get_device_variant()
        
        # Collect various device information
        info_commands = {
            'model': "cat /etc/device_info 2>/dev/null || echo 'N/A'",
            'firmware': "cat /etc/openwrt_release 2>/dev/null || cat /etc/os-release 2>/dev/null || echo 'N/A'",
            'kernel': "uname -r",
            'uptime': "uptime",
            'memory': "free -h | grep Mem:",
            'cpu': "cat /proc/cpuinfo | grep 'model name' | head -1"
        }
        
        for key, cmd in info_commands.items():
            stdout, _, _ = self.execute_command(cmd)
            
            if key == 'model':
                self.device_info.model = stdout.strip()
            elif key == 'firmware':
                self.device_info.firmware_version = stdout.strip()
            elif key == 'kernel':
                self.device_info.kernel_version = stdout.strip()
            elif key == 'uptime':
                self.device_info.uptime = stdout.strip()
            elif key == 'memory' and stdout:
                parts = stdout.split()
                if len(parts) >= 3:
                    self.device_info.memory_total = parts[1]
                    self.device_info.memory_free = parts[3] if len(parts) > 3 else "N/A"
            elif key == 'cpu':
                self.device_info.cpu_info = stdout.strip() or "N/A"
        
        return self.device_info
    
    def fwb_configs(self) -> str:
        """Collect FWB configurations from /etc/config/"""
        self.logger.info("-------------Collecting FWB Configs-------------")
        
        config_folder = self.local_path / f"{self.variant_connected}_configs_folder"
        config_folder.mkdir(exist_ok=True)
        
        output_file_path = config_folder / f'combined_configs_{self.variant_connected}.txt'
        
        start_time = datetime.datetime.now()
        
        with open(output_file_path, 'w') as output_file:
            stdout, stderr, code = self.execute_command("ls /etc/config/")
            
            if code != 0:
                self.logger.error(f"Error listing config files: {stderr}")
                output_file.write(f"Error listing config files: {stderr}\n")
            else:
                configs = stdout.splitlines()
                self.logger.info(f"Files in /etc/config: {configs}")
                
                for config in configs:
                    if config and not config.startswith('.'):
                        self.logger.info(f"Reading contents of {config}")
                        content, stderr, code = self.execute_command(f"cat /etc/config/{config}")
                        
                        if code == 0 and content:
                            output_file.write(f"----Contents of {config}----\n")
                            output_file.write(content)
                            output_file.write("\n\n")
                        elif stderr:
                            self.logger.error(f"Error reading {config}: {stderr}")
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create test result
        result = TestResult(
            test_name="FWB_Configs_Collection",
            test_category="Configuration Collection",
            status="PASS" if code == 0 else "FAIL",
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            output=f"Collected configs from /etc/config/",
            error=""
        )
        self.test_results.append(result)
        
        return str(output_file_path)
    
    def trees_dump(self) -> str:
        """Collect transformer-cli tree dumps"""
        self.logger.info("-------------Collecting Trees-------------")
        
        trees_dump_folder = self.local_path / f"{self.variant_connected}_trees_dump_folder"
        trees_dump_folder.mkdir(exist_ok=True)
        
        output_file_path = trees_dump_folder / f'trees_dump_{self.variant_connected}.txt'
        
        start_time = datetime.datetime.now()
        
        with open(output_file_path, 'w') as output_file:
            trees = ["Device.", "rpc.", "sys.", "uci."]
            
            for tree in trees:
                command = f"transformer-cli get {tree}"
                self.logger.info(f"Executing: {command}")
                
                stdout, stderr, code = self.execute_command(command, timeout=120)
                
                if code == 0 and stdout:
                    self.logger.info(f"Got data for tree: {tree}")
                    output_file.write(f"Tree: {tree}\n{stdout}\n\n")
                else:
                    error_msg = stderr or "No data returned"
                    self.logger.error(f"Error for tree {tree}: {error_msg}")
                    output_file.write(f"Tree: {tree}\nError: {error_msg}\n\n")
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = TestResult(
            test_name="Transformer_Trees_Dump",
            test_category="Trees Collection",
            status="PASS",
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            output=f"Dumped {len(trees)} trees",
            error=""
        )
        self.test_results.append(result)
        
        return str(output_file_path)
    
    def collecting_data(self) -> str:
        """Collect system data"""
        self.logger.info("-------------Collecting Data-------------")
        
        start_time = datetime.datetime.now()
        
        # Check/create remote directory
        check_cmd = f"if [ ! -d /tmp/{self.variant_connected} ]; then echo 'Not Exists'; else echo 'Exists already'; fi"
        stdout, _, _ = self.execute_command(check_cmd)
        
        if 'Not Exists' in stdout:
            self.execute_command(f"mkdir -p /tmp/{self.variant_connected}")
            self.logger.info(f"{self.variant_connected} folder created in /tmp")
        
        # Execute data collection commands
        data_commands = [
            f"lsmod > /tmp/{self.variant_connected}/lsmod.txt",
            f"uci show > /tmp/{self.variant_connected}/uci.txt",
            f"df -h > /tmp/{self.variant_connected}/df.txt",
            f"mount > /tmp/{self.variant_connected}/mount.txt",
            f"ifconfig > /tmp/{self.variant_connected}/ifconfig.txt",
            f"brctl show > /tmp/{self.variant_connected}/brctl.txt",
            f"route -n > /tmp/{self.variant_connected}/route.txt",
            f"iptables -S -t filter > /tmp/{self.variant_connected}/iptables-filter.txt",
            f"iptables -S -t nat > /tmp/{self.variant_connected}/iptables-nat.txt",
            f"iptables -S -t mangle > /tmp/{self.variant_connected}/iptables-mangle.txt",
            f"iptables -S -t raw > /tmp/{self.variant_connected}/iptables-raw.txt",
            f"ps > /tmp/{self.variant_connected}/ps.txt",
            f"find / -type f -name '*.conf' 2>/dev/null | head -1000 > /tmp/{self.variant_connected}/conf_files.txt"
        ]
        
        for cmd in data_commands:
            self.logger.info(f"Executing: {cmd}")
            self.execute_command(cmd)
        
        # Create local folder and download files
        data_folder = self.local_path / f"{self.variant_connected}_data_folder"
        if data_folder.exists():
            shutil.rmtree(data_folder)
        data_folder.mkdir(exist_ok=True)
        
        # Download using native SCP
        conn = self.connection_pool.get_connection()
        success = conn.download_directory(f"/tmp/{self.variant_connected}", str(data_folder))
        self.connection_pool.return_connection(conn)
        
        if success:
            self.logger.info(f"Data downloaded to: {data_folder}")
            status = "PASS"
            error = ""
        else:
            self.logger.error("Failed to download data")
            status = "FAIL"
            error = "SCP download failed"
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = TestResult(
            test_name="System_Data_Collection",
            test_category="Data Collection",
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            output=f"Collected {len(data_commands)} data files",
            error=error
        )
        self.test_results.append(result)
        
        return str(data_folder)
    
    def ubus_data(self) -> str:
        """Discover and test UBUS calls"""
        self.logger.info("-------------Collecting UBUS Data-------------")
        
        ubus_folder_path = self.local_path / f"{self.variant_connected}_ubus_folder"
        ubus_folder_path.mkdir(exist_ok=True)
        
        output_file_path = ubus_folder_path / f'combined_ubus_{self.variant_connected}.txt'
        
        start_time = datetime.datetime.now()
        
        # UBUS configuration
        check_words = ["get", "list", "status", "dump", "probe", "firmware_upgrade", 
                      "devices", "capabilities", "update", "info", "result", "configuration"]
        
        skip_list = ["hpna get", "network.interface.6rd status", "to get", "intercept status"]
        skip_keywords = {"mmpbx", "mmb", "mmdbd"}
        wifi_keywords = {"wireless.accesspoint.station get", "wireless.station get"}
        
        # Special UBUS values mapping
        ubus_values = {
            "igmpproxy.interface get": "igmpproxy.interface dump"
        }
        
        # Discover UBUS calls
        concatenated_results = set()
        
        # Search for UBUS calls
        grep_commands = [
            'cd /usr/; grep ":call" -ri * 2>/dev/null | head -100',
            'cd /etc/; grep ":call" -ri * 2>/dev/null | head -100',
            'cd /usr/; grep "ubus call" -ri * 2>/dev/null | head -100'
        ]
        
        for cmd in grep_commands:
            stdout, _, _ = self.execute_command(cmd, timeout=60)
            if stdout:
                # Parse UBUS calls
                lines = stdout.splitlines()[:50]  # Limit to 50 lines
                for line in lines:
                    # Extract UBUS calls using regex
                    matches = re.findall(r'ubus call (\S+) (\S+)', line)
                    for ubus_obj, method in matches:
                        if any(word in method for word in check_words):
                            concatenated_results.add(f"{ubus_obj} {method}")
        
        # Execute UBUS calls
        failed_ubus, skipped_ubus, succeeded_ubus = set(), set(), set()
        
        with open(output_file_path, 'w') as output_file:
            for i, result in enumerate(concatenated_results):
                if i >= 30:  # Limit to 30 calls
                    break
                
                # Skip checks
                should_skip = False
                if any(keyword in result for keyword in skip_keywords):
                    skipped_ubus.add(result)
                    output_file.write(f"ubus = {result}\nSkipped (MMPBX)\n{'='*40}\n")
                    should_skip = True
                elif any(keyword in result for keyword in wifi_keywords):
                    skipped_ubus.add(result)
                    output_file.write(f"ubus = {result}\nSkipped (WiFi required)\n{'='*40}\n")
                    should_skip = True
                
                if should_skip:
                    continue
                
                # Execute UBUS call
                if result in ubus_values:
                    command = f'ubus call {ubus_values[result]}'
                else:
                    command = f'ubus call {result}'
                
                stdout, stderr, code = self.execute_command(command, timeout=10)
                
                if code != 0 or stderr:
                    failed_ubus.add(result)
                    output_file.write(f"ubus = {result}\nError: {stderr}\n{'='*40}\n")
                else:
                    succeeded_ubus.add(result)
                    output_file.write(f"ubus = {result}\n{stdout[:500]}\n{'='*40}\n")
            
            # Write summary
            summary = f"""
{'='*50}
Execution Summary:
Total UBUS Calls Found: {len(concatenated_results)}
Succeeded: {len(succeeded_ubus)}
Failed: {len(failed_ubus)}
Skipped: {len(skipped_ubus)}
{'='*50}
"""
            output_file.write(summary)
            self.logger.info(summary)
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = TestResult(
            test_name="UBUS_Call_Testing",
            test_category="UBUS Testing",
            status="PASS" if len(failed_ubus) == 0 else "WARN",
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            output=f"Succeeded: {len(succeeded_ubus)}, Failed: {len(failed_ubus)}",
            error=""
        )
        self.test_results.append(result)
        
        return str(output_file_path)
    
    def test_tr181_data_model(self) -> List[TestResult]:
        """Test TR-181 data model parameters"""
        self.logger.info("Testing TR-181 data model...")
        test_results = []
        
        # Get TR-181 parameters from config
        tr181_params = []
        for category, params in self.config.get('tr181_parameters', {}).items():
            tr181_params.extend(params)
        
        # Default parameters if none in config
        if not tr181_params:
            tr181_params = [
                "Device.DeviceInfo.Manufacturer",
                "Device.DeviceInfo.ModelName",
                "Device.DeviceInfo.SoftwareVersion",
                "Device.LAN.IPAddress",
                "Device.WiFi.Radio.1.Enable"
            ]
        
        for param in tr181_params[:10]:  # Limit to 10
            start_time = datetime.datetime.now()
            
            # Try different TR-181 access methods
            commands = [
                f"dmcli eRT getv {param}",
                f"tr181 get {param}",
                f"ccsp_bus_client getv {param}",
                f"ubus call tr181 get '{{\"parameter\": \"{param}\"}}'",
                f"transformer-cli get {param}"
            ]
            
            success = False
            output = ""
            error = ""
            
            for cmd in commands:
                stdout, stderr, code = self.execute_command(cmd, timeout=5)
                if code == 0 and stdout:
                    output = stdout[:200]
                    success = True
                    break
                else:
                    error = stderr or "Command failed"
            
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = TestResult(
                test_name=f"TR181_{param.replace('.', '_')}",
                test_category="TR-181 Data Model",
                status="PASS" if success else "FAIL",
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                output=output,
                error=error,
                metadata={"parameter": param}
            )
            
            test_results.append(result)
            self.test_results.append(result)
        
        return test_results
    
    def test_uci_configuration(self) -> List[TestResult]:
        """Test UCI and UCI-CLI configuration"""
        self.logger.info("Testing UCI configuration...")
        test_results = []
        
        uci_configs = ["system", "network", "wireless", "firewall", "dhcp"]
        
        for config in uci_configs:
            # Test UCI show
            start_time = datetime.datetime.now()
            stdout, stderr, code = self.execute_command(f"uci show {config} 2>/dev/null | head -20")
            end_time = datetime.datetime.now()
            
            result = TestResult(
                test_name=f"UCI_{config}_show",
                test_category="UCI Configuration",
                status="PASS" if code == 0 else "FAIL",
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds(),
                output=stdout[:500],
                error=stderr
            )
            test_results.append(result)
            self.test_results.append(result)
            
            # Test UCI-CLI if available
            start_time = datetime.datetime.now()
            stdout, stderr, code = self.execute_command(f"uci-cli get {config} 2>/dev/null | head -20")
            end_time = datetime.datetime.now()
            
            if code == 0 or "not found" not in stderr.lower():
                result = TestResult(
                    test_name=f"UCI-CLI_{config}_get",
                    test_category="UCI-CLI",
                    status="PASS" if code == 0 else "FAIL",
                    start_time=start_time,
                    end_time=end_time,
                    duration=(end_time - start_time).total_seconds(),
                    output=stdout[:500],
                    error=stderr
                )
                test_results.append(result)
                self.test_results.append(result)
        
        return test_results
    
    def test_linux_tools(self) -> List[TestResult]:
        """Test various Linux tools and system functionality"""
        self.logger.info("Testing Linux tools...")
        test_results = []
        
        linux_tests = [
            ("uptime", "System Uptime"),
            ("free -m", "Memory Usage"),
            ("df -h", "Disk Usage"),
            ("ps aux | head -10", "Top Processes"),
            ("netstat -tuln | head -10", "Network Connections"),
            ("lsmod | head -10", "Loaded Modules"),
            ("dmesg | tail -20", "Kernel Messages"),
            ("cat /proc/cpuinfo | grep processor | wc -l", "CPU Count"),
            ("cat /proc/meminfo | grep MemTotal", "Total Memory"),
            ("ip addr show", "Network Interfaces")
        ]
        
        for command, description in linux_tests:
            start_time = datetime.datetime.now()
            stdout, stderr, code = self.execute_command(command, timeout=10)
            end_time = datetime.datetime.now()
            
            result = TestResult(
                test_name=f"Linux_{description.replace(' ', '_')}",
                test_category="Linux Tools",
                status="PASS" if code == 0 else "FAIL",
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds(),
                output=stdout[:500],
                error=stderr,
                metadata={"command": command}
            )
            
            test_results.append(result)
            self.test_results.append(result)
        
        return test_results
    
    def enable_and_collect_logs(self) -> str:
        """Enable logging and collect device logs"""
        self.logger.info("Enabling and collecting device logs...")
        
        log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        device_log_dir = LOG_DIR / f"device_logs_{log_timestamp}"
        device_log_dir.mkdir(exist_ok=True)
        
        # Enable verbose logging
        log_commands = [
            "uci set system.@system[0].log_size=1024 2>/dev/null",
            "uci set system.@system[0].log_level=debug 2>/dev/null",
            "uci commit system 2>/dev/null",
            "/etc/init.d/log restart 2>/dev/null || /etc/init.d/syslog restart 2>/dev/null"
        ]
        
        for cmd in log_commands:
            self.execute_command(cmd)
        
        # Wait for logs to accumulate
        time.sleep(3)
        
        # Collect logs
        log_commands = [
            ("logread", "system.log"),
            ("dmesg", "kernel.log"),
            ("cat /var/log/messages 2>/dev/null || echo 'No messages log'", "messages.log")
        ]
        
        for cmd, filename in log_commands:
            stdout, _, _ = self.execute_command(cmd)
            if stdout and stdout != "No messages log":
                log_file = device_log_dir / filename
                with open(log_file, 'w') as f:
                    f.write(stdout)
                self.logger.info(f"Collected log: {filename}")
        
        return str(device_log_dir)
    
    def run_parallel_tests(self) -> List[TestResult]:
        """Run tests in parallel for optimal performance"""
        self.logger.info(f"Starting parallel test execution with {self.max_workers} workers...")
        
        test_functions = [
            self.test_tr181_data_model,
            self.test_uci_configuration,
            self.test_linux_tools,
        ]
        
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, 4)) as executor:
            future_to_test = {
                executor.submit(test_func): test_func.__name__ 
                for test_func in test_functions
            }
            
            for future in concurrent.futures.as_completed(future_to_test):
                test_name = future_to_test[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    self.logger.info(f"Completed: {test_name} - {len(results)} tests")
                except Exception as e:
                    self.logger.error(f"Test {test_name} failed: {e}")
        
        return all_results
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        summary = {
            "total_tests": len(self.test_results),
            "passed": sum(1 for r in self.test_results if r.status == "PASS"),
            "failed": sum(1 for r in self.test_results if r.status == "FAIL"),
            "warnings": sum(1 for r in self.test_results if r.status == "WARN"),
            "errors": sum(1 for r in self.test_results if r.status == "ERROR"),
            "pass_rate": 0.0,
            "total_duration": 0.0,
            "test_categories": defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0}),
            "device_info": self.device_info.__dict__,
            "test_start_time": min(r.start_time for r in self.test_results) if self.test_results else None,
            "test_end_time": max(r.end_time for r in self.test_results) if self.test_results else None,
        }
        
        if self.test_results:
            summary["pass_rate"] = (summary["passed"] / summary["total_tests"]) * 100
            summary["total_duration"] = sum(r.duration for r in self.test_results)
            
            for result in self.test_results:
                cat = result.test_category
                summary["test_categories"][cat]["total"] += 1
                if result.status == "PASS":
                    summary["test_categories"][cat]["passed"] += 1
                elif result.status == "FAIL":
                    summary["test_categories"][cat]["failed"] += 1
        
        return summary
    
    def generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate HTML report"""
        self.logger.info("Generating HTML report...")
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_file = REPORT_DIR / f"test_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Purple OS Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .header { background-color: #6a1b9a; color: white; padding: 20px; border-radius: 5px; }
                .summary { background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric { display: inline-block; margin: 10px 20px; }
                .metric-value { font-size: 24px; font-weight: bold; }
                .metric-label { color: #666; }
                .passed { color: #4caf50; }
                .failed { color: #f44336; }
                .warning { color: #ff9800; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th { background-color: #6a1b9a; color: white; padding: 10px; text-align: left; }
                td { padding: 8px; border-bottom: 1px solid #ddd; }
                .test-pass { background-color: #e8f5e9; }
                .test-fail { background-color: #ffebee; }
                .test-warn { background-color: #fff3e0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Purple OS Automated Test Report</h1>
                <p>Generated: {{ timestamp }}</p>
                <p>Device: {{ device_info.variant }}</p>
            </div>
            
            <div class="summary">
                <h2>Test Summary</h2>
                <div class="metric">
                    <div class="metric-value">{{ total_tests }}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
                <div class="metric">
                    <div class="metric-value passed">{{ passed }}</div>
                    <div class="metric-label">Passed</div>
                </div>
                <div class="metric">
                    <div class="metric-value failed">{{ failed }}</div>
                    <div class="metric-label">Failed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ "%.1f"|format(pass_rate) }}%</div>
                    <div class="metric-label">Pass Rate</div>
                </div>
            </div>
            
            <div class="summary">
                <h2>Test Results by Category</h2>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Total</th>
                        <th>Passed</th>
                        <th>Failed</th>
                        <th>Pass Rate</th>
                    </tr>
                    {% for cat, stats in test_categories.items() %}
                    <tr>
                        <td>{{ cat }}</td>
                        <td>{{ stats.total }}</td>
                        <td class="passed">{{ stats.passed }}</td>
                        <td class="failed">{{ stats.failed }}</td>
                        <td>{{ "%.1f"|format((stats.passed / stats.total * 100) if stats.total > 0 else 0) }}%</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="summary">
                <h2>Detailed Test Results</h2>
                <table>
                    <tr>
                        <th>Test Name</th>
                        <th>Category</th>
                        <th>Status</th>
                        <th>Duration (s)</th>
                    </tr>
                    {% for result in test_results %}
                    <tr class="test-{{ result.status.lower() }}">
                        <td>{{ result.test_name }}</td>
                        <td>{{ result.test_category }}</td>
                        <td>{{ result.status }}</td>
                        <td>{{ "%.3f"|format(result.duration) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(
            timestamp=timestamp,
            device_info=summary["device_info"],
            total_tests=summary["total_tests"],
            passed=summary["passed"],
            failed=summary["failed"],
            pass_rate=summary["pass_rate"],
            test_categories=dict(summary["test_categories"]),
            test_results=self.test_results
        )
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {report_file}")
        return str(report_file)
    
    def generate_text_dashboard(self, summary: Dict[str, Any]) -> str:
        """Generate text-based dashboard"""
        self.logger.info("Generating text dashboard...")
        
        dashboard_file = DASHBOARD_DIR / f"dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(dashboard_file, 'w') as f:
            f.write(f"Purple OS Test Dashboard - {self.variant_connected}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device_info.variant}\n")
            f.write(f"Firmware: {self.device_info.firmware_version[:50]}...\n")
            f.write("\n")
            f.write("Test Summary\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Tests: {summary['total_tests']}\n")
            f.write(f"Passed: {summary['passed']} ({summary['pass_rate']:.1f}%)\n")
            f.write(f"Failed: {summary['failed']}\n")
            f.write(f"Warnings: {summary.get('warnings', 0)}\n")
            f.write(f"Duration: {summary['total_duration']:.2f}s\n")
            f.write("\n")
            f.write("Category Summary\n")
            f.write("-" * 30 + "\n")
            
            for cat, stats in summary['test_categories'].items():
                pass_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
                f.write(f"{cat:.<30} {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)\n")
        
        self.logger.info(f"Dashboard generated: {dashboard_file}")
        return str(dashboard_file)
    
    def run_complete_test_suite(self):
        """Run the complete test suite"""
        self.logger.info("=" * 80)
        self.logger.info("Starting Purple OS Automated Testing Suite (Native SSH)")
        self.logger.info("=" * 80)
        
        try:
            # Get device variant and create folders
            self.get_device_variant()
            
            # Collect device information
            self.collect_device_info()
            
            # Run legacy integrated tests
            self.logger.info("Running integrated test suite...")
            config_path = self.fwb_configs()
            trees_path = self.trees_dump()
            data_path = self.collecting_data()
            ubus_path = self.ubus_data()
            
            # Enable and collect logs
            log_dir = self.enable_and_collect_logs()
            self.logger.info(f"Device logs collected in: {log_dir}")
            
            # Run additional tests in parallel
            self.run_parallel_tests()
            
            # Generate test summary
            summary = self.generate_test_summary()
            
            # Generate reports
            html_report = self.generate_html_report(summary)
            dashboard = self.generate_text_dashboard(summary)
            
            # Save summary as JSON
            summary_file = REPORT_DIR / f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Print summary
            self.logger.info("=" * 80)
            self.logger.info("TEST EXECUTION SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"Device Variant: {self.variant_connected}")
            self.logger.info(f"Total Tests: {summary['total_tests']}")
            self.logger.info(f"Passed: {summary['passed']} ({summary['pass_rate']:.1f}%)")
            self.logger.info(f"Failed: {summary['failed']}")
            self.logger.info(f"Duration: {summary['total_duration']:.2f} seconds")
            self.logger.info("-" * 80)
            self.logger.info("Output Files:")
            self.logger.info(f"Config folder: {config_path}")
            self.logger.info(f"Trees dump: {trees_path}")
            self.logger.info(f"Data collection: {data_path}")
            self.logger.info(f"UBUS data: {ubus_path}")
            self.logger.info(f"HTML Report: {html_report}")
            self.logger.info(f"Dashboard: {dashboard}")
            self.logger.info(f"Summary JSON: {summary_file}")
            self.logger.info(f"Logs: {log_dir}")
            self.logger.info("=" * 80)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Purple OS Testing Framework - Native SSH')
    parser.add_argument('--device-ip', default=DEVICE_IP, help='Device IP address')
    parser.add_argument('--username', default=DEVICE_USERNAME, help='SSH username')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    print(f"Connecting to device at {args.device_ip}...")
    
    try:
        framework = PurpleOSTestFramework(
            device_ip=args.device_ip,
            username=args.username,
            max_workers=args.workers
        )
        
        summary = framework.run_complete_test_suite()
        
        # Exit with appropriate code
        sys.exit(0 if summary['failed'] == 0 else 1)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
