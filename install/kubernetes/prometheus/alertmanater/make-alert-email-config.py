import base64
# alert 报警的secret

# 换行都会变换base64编码后的格式
config='''global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/your_slack_api_token'
  smtp_smarthost: 'smtp.exmail.qq.com:465'
  smtp_from: 'xxxx@tencent.com'
  smtp_hello: 'tencent.com'
  smtp_auth_username: 'xxxx@tencent.com'
  smtp_auth_password: 'xxxxxxxxx'
  smtp_require_tls: false
templates:
- '/etc/alertmanager/template/*.tmpl'
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 1h
  receiver: 'null'
  routes:
  - match:
      namespace: infra
    receiver: 'default'

inhibit_rules:
- source_match:
    severity: 'critical'
  target_match:
    severity: 'warning'
  # Apply inhibition if the alertname is the same.
  equal: ['alertname', 'cluster', 'service']
receivers:
- name: 'default'
  slack_configs:
  - channel: '#your_slack_channel'
    title: '[{{ .Status | toUpper }}{{ if eq .Status "firing" }}:{{ .Alerts.Firing | len }}{{ end }}] Prometheus Event Notification'
    text: >-
        {{ range .Alerts }}
           *Alert:* {{ .Annotations.summary }} - `{{ .Labels.severity }}`
          *Description:* {{ .Annotations.description }}
          *Graph:* <{{ .GeneratorURL }}|:chart_with_upwards_trend:> *Runbook:* <{{ .Annotations.runbook }}|:spiral_note_pad:>
          *Details:*
          {{ range .Labels.SortedPairs }} • *{{ .Name }}:* `{{ .Value }}`
          {{ end }}
        {{ end }}
    send_resolved: false
  email_configs:
  - to: 'xxxx@tencent.com'
    send_resolved: true
- name: 'null'
'''

base64str = base64.b64encode(bytes(config,encoding='utf-8'))
print(str(base64str,encoding='utf-8'))
