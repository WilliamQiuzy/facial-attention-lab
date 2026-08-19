# AFLFP access packet

Status as of 2026-08-05: application evidence supports `application_sent` on
2026-06-18 and `reply_received` on 2026-06-20. It does not establish
`access_granted`, approval, a download link, or local acquisition.

## Official sources and terms

- [Official AFLFP README](https://github.com/Yifan313/AFLFP/blob/main/README.md)
- [University of Portsmouth dataset page](https://researchportal.port.ac.uk/en/datasets/aflfp-database/)
- [Official End User License Agreement](https://github.com/Yifan313/AFLFP/blob/main/End_User_License_Agreement.pdf)

The README describes AFLFP as publicly available for non-commercial academic
research through an application. It requires an application from a valid
academic or institutional email account, sent to the official recipient and
copied to the official secondary contact, with the End User License Agreement
attached. The reviewed EULA SHA-256 is
`15b4cf3beb1d9ea4da519b267d31deeb9aa9c22db8beca1ee33e4fdced8fcca3`.

The EULA requires the recipient not to be a student and to be eligible as either
a full-time faculty researcher or an organization employee. It also
requires the recipient's signature. Software must not determine eligibility,
sign or accept the EULA, populate a person's attestation, or send the message.
Those actions remain with the eligible researcher using an institutional email.

The EULA prohibits the recipient from making any further copy of, publishing,
or distributing any part of the database. Only academic analyses and results
may be published. Images included in a publication or presentation are limited
to images of the 22 subjects specifically listed in the EULA; that limited image
permission does not permit copying, publishing, or distributing any other part
of the database.

## Evidence boundary

The mailbox review was used only to establish the two aggregate dates above.
No message IDs, mailbox addresses beyond the public application contacts,
private correspondence, tokens, attachments, or secrets are recorded here. A
received reply is not treated as approval or access. The current access state is
therefore `not_established`.

## Exact official application text — draft only

This reproduces the application text required by the official README. It is not
a sent message and does not accept the EULA on anyone's behalf.

```text
To: hui.yu@glasgow.ac.uk
Cc: xiayifan@sdu.edu.cn
Subject: Application to download the AFLFP Database

Name: <your first and last name>
Institution: <where you work>
Email: <must be the email at the above mentioned institution>

I have read and agree to the terms specified in the End User License Agreement.
This database will only be used for research purposes.
```

Before any future transmission, an eligible researcher must personally verify
the current README and EULA, complete the placeholders, sign the agreement, and
attach `End_User_License_Agreement.pdf` before authorizing sending. Do not resend
merely because this draft exists, and do not record `access_granted` without
explicit grant evidence.
