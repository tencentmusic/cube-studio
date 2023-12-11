import socket
import requests

def get_public_ip():
    try:
        response = requests.get('https://api64.ipify.org?format=json')
        response.raise_for_status()
        ip = response.json()['ip']
        return ip
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None



def get_ips(domain):
    try:
        ips = socket.getaddrinfo(domain, None)
        unique_ips = set()
        for ip in ips:
            unique_ips.add(ip[4][0])
        return list(unique_ips)
    except socket.gaierror as e:
        print("Error: ", e)
        return []

