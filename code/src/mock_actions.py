import time

# ✅ Mock functions to simulate server tasks
def mock_restart_service(service_name: str):
    return f"✅ Server {service_name} has been restarted successfully!"

def mock_get_service_status(service_name: str):
    return f"ℹ️ The status of server {service_name} is: Running"

def mock_get_system_logs(service_name: str):
    return f"""
    2025-03-24 10:15:32.123 INFO  [{service_name}] Application started successfully on port 8080
    2025-03-24 10:15:35.456 WARN  [{service_name}] Slow response time detected: 1200ms for request /api/orders
    2025-03-24 10:15:37.789 ERROR [{service_name}] Exception occurred while processing request /api/users
    java.lang.NullPointerException: Cannot read property 'name' of null
        at com.example.Controller.getUserDetails(Controller.java:45)
        at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
        at org.springframework.web.servlet.FrameworkServlet.service(FrameworkServlet.java:882)
    2025-03-24 10:15:40.567 DEBUG [{service_name}] Executing query: SELECT * FROM users WHERE id=?
    """

def mock_send_email(service_name: str):
    return f"""
    Email Sent Successfully!
    
    Subject: Alert - Issue Detected on {service_name}
    
    Hi Team,
    
    An issue has been detected on {service_name}. Please check the system logs for further details.
    
    Timestamp: 2025-03-24 10:20:00
    Error Summary: High CPU usage detected (90%) and multiple failed login attempts.
    
    Recommended Action:
    - Check system performance metrics.
    - Investigate unauthorized login attempts.
    
    Regards,
    Automated Monitoring System
    """