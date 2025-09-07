import * as pulumi from "@pulumi/pulumi";
import * as k8s from "@pulumi/kubernetes";

/**
 * קונפיג סודי – לא נשמר בגיט.
 * הגדר עם:
 * pulumi config set --secret alerts:slackWebhook 'https://hooks.slack.com/services/...'
 * pulumi config set --secret alerts:teamsWebhook  'https://outlook.office.com/webhook/...'
 * pulumi config set --secret alerts:emailTo       'oncall@example.com'
 * pulumi config set        alerts:emailFrom       'alerts@example.com'
 * pulumi config set        alerts:smtpHost        'smtp.example.com:587'
 * pulumi config set --secret alerts:smtpUser      'username'
 * pulumi config set --secret alerts:smtpPass      'password'
 * (ניתן להשמיט ערוצים שלא בשימוש – gating יבוטל עם values.alerts.* ריקים)
 */

const cfg = new pulumi.Config("alerts");
const slackWebhook = cfg.getSecret("slackWebhook") || pulumi.secret("");
const teamsWebhook = cfg.getSecret("teamsWebhook") || pulumi.secret("");
const emailTo      = cfg.get("emailTo") || "";
const emailFrom    = cfg.get("emailFrom") || "";
const smtpHost     = cfg.get("smtpHost") || "";
const smtpUser     = cfg.getSecret("smtpUser") || pulumi.secret("");
const smtpPass     = cfg.getSecret("smtpPass") || pulumi.secret("");

// ArgoCD Application שמנהל את umbrella-prod
const appName = "imu-umbrella-prod";
const appNs   = "argocd";

// נמשוך את ה־Application ונבצע Patch אסטרטגי ל־Helm values (parameters)
const app = new k8s.apiextensions.CustomResource(appName, {
  apiVersion: "argoproj.io/v1alpha1",
  kind: "Application",
  metadata: { name: appName, namespace: appNs },
  spec: {
    // שומרים על הערכים הקיימים; אנחנו מזריקים parameters
    // פאטצ' ב־apply: merge – נניח שה-umbrella כבר קיים
  },
}, { ignoreChanges: ["spec"], replaceOnChanges: ["metadata"] });

// יצירת Resource patch דרך ServerSide Apply (SSA)
const patch = new k8s.apiextensions.CustomResource(`${appName}-patch`, {
  apiVersion: "argoproj.io/v1alpha1",
  kind: "Application",
  metadata: { name: appName, namespace: appNs },
  spec: {
    source: {
      helm: {
        // מוסיפים/מעדכנים פרמטרים (write-back בצד ArgoCD/Helm בזמן הרינדור)
        parameters: [
          { name: "alerts.slack.webhook", value: slackWebhook },
          { name: "alerts.teams.webhook", value: teamsWebhook },
          { name: "alerts.email.to",      value: emailTo },
          { name: "alerts.email.from",    value: emailFrom },
          { name: "alerts.email.smarthost", value: smtpHost },
          { name: "alerts.email.authUsername", value: smtpUser },
          { name: "alerts.email.authPassword", value: smtpPass }
        ]
      }
    }
  }
}, {
  dependsOn: app,
  customTimeouts: { create: "2m", update: "2m" }
});

export const patched = patch.metadata.name;
