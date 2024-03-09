import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(sender_address, receiver_address, password, filename, logger):

    body = "メールの本文"

    # メールオブジェクトの作成
    msg = MIMEMultipart()
    msg["From"] = sender_address
    msg["To"] = receiver_address
    msg["Subject"] = filename

    # メール本文の追加
    msg.attach(MIMEText(body, "plain"))

    # 添付ファイルの追加
    attachment = open(filename, "rb")

    part = MIMEBase("application", "octet-stream")
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition", "attachment; filename= %s" % filename
    )

    msg.attach(part)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_address, password)
        text = msg.as_string()
        server.sendmail(sender_address, receiver_address, text)
        server.quit()
        logger.info("メールが正常に送信されました。")
    except Exception as e:
        logger.info(f"メール送信中にエラーが発生しました: {e}")
    finally:
        attachment.close()
