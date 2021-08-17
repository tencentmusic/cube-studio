import base64
# alert 报警的secret
# 换行都会变换base64编码后的格式
config='''
global:
  resolve_timeout: 5m
templates:
- '/etc/alertmanager/template/*.tmpl'
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 1h
  receiver: 'null'
  routes:
  - match_re:
      namespace: ".*"
    receiver: 'default'

# 告警抑制。避免重复告警的。
# https://yunlzheng.gitbook.io/prometheus-book/parti-prometheus-ji-chu/alert/alert-manager-inhibit
inhibit_rules:
- source_match:
    severity: 'critical'
  target_match:
    severity: 'warning'
  # Apply inhibition if the alertname is the same.
  equal: ['alertname', 'cluster', 'service']
receivers:

- name: 'default'
  webhook_configs:
  - send_resolved: true
    url: 'http://xx.xx.xx.xx/'

- name: 'null'
'''

base64str = base64.b64encode(bytes(config,encoding='utf-8'))
print(str(base64str,encoding='utf-8'))
