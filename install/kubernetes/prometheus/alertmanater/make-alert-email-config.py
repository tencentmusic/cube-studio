import base64
# alert 报警的secret
# base64str = "Z2xvYmFsOgogIHJlc29sdmVfdGltZW91dDogNW0KICBzbGFja19hcGlfdXJsOiAnaHR0cHM6Ly9ob29rcy5zbGFjay5jb20vc2VydmljZXMveW91cl9zbGFja19hcGlfdG9rZW4nCiAgc210cF9zbWFydGhvc3Q6ICd5b3VyX3NtdHBfc21hcnRob3N0OjU4NycKICBzbXRwX2Zyb206ICd5b3VyX3NtdHBfZnJvbScKICBzbXRwX2F1dGhfdXNlcm5hbWU6ICd5b3VyX3NtdHBfdXNlcicKICBzbXRwX2F1dGhfcGFzc3dvcmQ6ICd5b3VyX3NtdHBfcGFzcycKdGVtcGxhdGVzOgotICcvZXRjL2FsZXJ0bWFuYWdlci90ZW1wbGF0ZS8qLnRtcGwnCnJvdXRlOgogIGdyb3VwX2J5OiBbJ2FsZXJ0bmFtZScsICdjbHVzdGVyJywgJ3NlcnZpY2UnXQogIGdyb3VwX3dhaXQ6IDMwcwogIGdyb3VwX2ludGVydmFsOiA1bQogIHJlcGVhdF9pbnRlcnZhbDogMWgKICByZWNlaXZlcjogZGVmYXVsdC1yZWNlaXZlcgogIHJvdXRlczoKICAtIG1hdGNoOgogICAgICBhbGVydG5hbWU6IERlYWRNYW5zU3dpdGNoCiAgICByZWNlaXZlcjogJ251bGwnCmluaGliaXRfcnVsZXM6Ci0gc291cmNlX21hdGNoOgogICAgc2V2ZXJpdHk6ICdjcml0aWNhbCcKICB0YXJnZXRfbWF0Y2g6CiAgICBzZXZlcml0eTogJ3dhcm5pbmcnCiAgIyBBcHBseSBpbmhpYml0aW9uIGlmIHRoZSBhbGVydG5hbWUgaXMgdGhlIHNhbWUuCiAgZXF1YWw6IFsnYWxlcnRuYW1lJywgJ2NsdXN0ZXInLCAnc2VydmljZSddCnJlY2VpdmVyczoKLSBuYW1lOiAnZGVmYXVsdC1yZWNlaXZlcicKICBzbGFja19jb25maWdzOgogIC0gY2hhbm5lbDogJyN5b3VyX3NsYWNrX2NoYW5uZWwnCiAgICB0aXRsZTogJ1t7eyAuU3RhdHVzIHwgdG9VcHBlciB9fXt7IGlmIGVxIC5TdGF0dXMgImZpcmluZyIgfX06e3sgLkFsZXJ0cy5GaXJpbmcgfCBsZW4gfX17eyBlbmQgfX1dIFByb21ldGhldXMgRXZlbnQgTm90aWZpY2F0aW9uJwogICAgdGV4dDogPi0KICAgICAgICB7eyByYW5nZSAuQWxlcnRzIH19CiAgICAgICAgICAgKkFsZXJ0Oioge3sgLkFubm90YXRpb25zLnN1bW1hcnkgfX0gLSBge3sgLkxhYmVscy5zZXZlcml0eSB9fWAKICAgICAgICAgICpEZXNjcmlwdGlvbjoqIHt7IC5Bbm5vdGF0aW9ucy5kZXNjcmlwdGlvbiB9fQogICAgICAgICAgKkdyYXBoOiogPHt7IC5HZW5lcmF0b3JVUkwgfX18OmNoYXJ0X3dpdGhfdXB3YXJkc190cmVuZDo+ICpSdW5ib29rOiogPHt7IC5Bbm5vdGF0aW9ucy5ydW5ib29rIH19fDpzcGlyYWxfbm90ZV9wYWQ6PgogICAgICAgICAgKkRldGFpbHM6KgogICAgICAgICAge3sgcmFuZ2UgLkxhYmVscy5Tb3J0ZWRQYWlycyB9fSDigKIgKnt7IC5OYW1lIH19OiogYHt7IC5WYWx1ZSB9fWAKICAgICAgICAgIHt7IGVuZCB9fQogICAgICAgIHt7IGVuZCB9fQogICAgc2VuZF9yZXNvbHZlZDogdHJ1ZQogIGVtYWlsX2NvbmZpZ3M6CiAgLSB0bzogJ3lvdXJfYWxlcnRfZW1haWxfYWRkcmVzcycKICAgIHNlbmRfcmVzb2x2ZWQ6IHRydWUKLSBuYW1lOiAnbnVsbCcK"
# base64str = "Z2xvYmFsOgogIHJlc29sdmVfdGltZW91dDogNW0KICBzbGFja19hcGlfdXJsOiAnaHR0cHM6Ly9ob29rcy5zbGFjay5jb20vc2VydmljZXMveW91cl9zbGFja19hcGlfdG9rZW4nCiAgc210cF9zbWFydGhvc3Q6ICdzbXRwLmV4bWFpbC5xcS5jb206NDY1JwogIHNtdHBfZnJvbTogJ2x1YW4ucGVuZ0BpbnRlbGxpZi5jb20nCiAgc210cF9oZWxsbzogJ2ludGVsbGlmLmNvbScKICBzbXRwX2F1dGhfdXNlcm5hbWU6ICdsdWFuLnBlbmdAaW50ZWxsaWYuY29tJwogIHNtdHBfYXV0aF9wYXNzd29yZDogJzFxYXoyd3N4I0VEQycKICBzbXRwX3JlcXVpcmVfdGxzOiBmYWxzZQp0ZW1wbGF0ZXM6Ci0gJy9ldGMvYWxlcnRtYW5hZ2VyL3RlbXBsYXRlLyoudG1wbCcKcm91dGU6CiAgZ3JvdXBfYnk6IFsnYWxlcnRuYW1lJywgJ2NsdXN0ZXInLCAnc2VydmljZSddCiAgZ3JvdXBfd2FpdDogMzBzCiAgZ3JvdXBfaW50ZXJ2YWw6IDVtCiAgcmVwZWF0X2ludGVydmFsOiAxaAogIHJlY2VpdmVyOiAnbnVsbCcKICByb3V0ZXM6CiAgLSBtYXRjaDoKICAgICAgbmFtZXNwYWNlOiBjbG91ZGFpLTIKICAgIHJlY2VpdmVyOiAnZGVmYXVsdC1yZWNlaXZlcicKaW5oaWJpdF9ydWxlczoKLSBzb3VyY2VfbWF0Y2g6CiAgICBzZXZlcml0eTogJ2NyaXRpY2FsJwogIHRhcmdldF9tYXRjaDoKICAgIHNldmVyaXR5OiAnd2FybmluZycKICAjIEFwcGx5IGluaGliaXRpb24gaWYgdGhlIGFsZXJ0bmFtZSBpcyB0aGUgc2FtZS4KICBlcXVhbDogWydhbGVydG5hbWUnLCAnY2x1c3RlcicsICdzZXJ2aWNlJ10KcmVjZWl2ZXJzOgotIG5hbWU6ICdkZWZhdWx0LXJlY2VpdmVyJwogIHNsYWNrX2NvbmZpZ3M6CiAgLSBjaGFubmVsOiAnI3lvdXJfc2xhY2tfY2hhbm5lbCcKICAgIHRpdGxlOiAnW3t7IC5TdGF0dXMgfCB0b1VwcGVyIH19e3sgaWYgZXEgLlN0YXR1cyAiZmlyaW5nIiB9fTp7eyAuQWxlcnRzLkZpcmluZyB8IGxlbiB9fXt7IGVuZCB9fV0gUHJvbWV0aGV1cyBFdmVudCBOb3RpZmljYXRpb24nCiAgICB0ZXh0OiA+LQogICAgICAgIHt7IHJhbmdlIC5BbGVydHMgfX0KICAgICAgICAgICAqQWxlcnQ6KiB7eyAuQW5ub3RhdGlvbnMuc3VtbWFyeSB9fSAtIGB7eyAuTGFiZWxzLnNldmVyaXR5IH19YAogICAgICAgICAgKkRlc2NyaXB0aW9uOioge3sgLkFubm90YXRpb25zLmRlc2NyaXB0aW9uIH19CiAgICAgICAgICAqR3JhcGg6KiA8e3sgLkdlbmVyYXRvclVSTCB9fXw6Y2hhcnRfd2l0aF91cHdhcmRzX3RyZW5kOj4gKlJ1bmJvb2s6KiA8e3sgLkFubm90YXRpb25zLnJ1bmJvb2sgfX18OnNwaXJhbF9ub3RlX3BhZDo+CiAgICAgICAgICAqRGV0YWlsczoqCiAgICAgICAgICB7eyByYW5nZSAuTGFiZWxzLlNvcnRlZFBhaXJzIH19IOKAoiAqe3sgLk5hbWUgfX06KiBge3sgLlZhbHVlIH19YAogICAgICAgICAge3sgZW5kIH19CiAgICAgICAge3sgZW5kIH19CiAgICBzZW5kX3Jlc29sdmVkOiBmYWxzZQogIGVtYWlsX2NvbmZpZ3M6CiAgLSB0bzogJ2x1YW4ucGVuZ0BpbnRlbGxpZi5jb20nCiAgICBzZW5kX3Jlc29sdmVkOiB0cnVlCi0gbmFtZTogJ251bGwnCg=="
# result = base64.b64decode(base64str)
#
# print(str(result,encoding='utf-8'))

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
