{{- define "cp.name" -}}
imu-control-plane
{{- end -}}

{{- define "cp.fullname" -}}
{{ include "cp.name" . }}-{{ .Release.Name }}
{{- end -}}