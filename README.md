# Dual arm manipulation

## Demos

<table width="100%">
  <tr>
    <td width="50%">
      <h3 align="center">8figure path (PID)</h3>
      <div align="center">
        <img src="https://private-user-images.githubusercontent.com/109598450/525376865-e1859f5f-7873-4921-badc-66ffa9a82055.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjU2Njg5MjEsIm5iZiI6MTc2NTY2ODYyMSwicGF0aCI6Ii8xMDk1OTg0NTAvNTI1Mzc2ODY1LWUxODU5ZjVmLTc4NzMtNDkyMS1iYWRjLTY2ZmZhOWE4MjA1NS5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMjEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTIxM1QyMzMwMjFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01N2IxODlhNjcxZTkxMTJlMTY5ZGJiNDlhZWMyNzY4Zjk0NDc0YmFmM2I4OWIwMjc5MmRmMDliMDcwYjExOTk3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.kFLFJFP4F68DNHuTumzOzAu3JKhAAvP0qMJAifIj0t8" alt="Demo 1" width="100%" />
      </div>
    </td>
    <td width="50%">
      <h3 align="center">Circular path (QP)</h3>
      <div align="center">
        <img src="https://private-user-images.githubusercontent.com/109598450/525383752-0bd4010f-02ec-4fe7-b1c3-788ac3912d69.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjU2Njg5NDQsIm5iZiI6MTc2NTY2ODY0NCwicGF0aCI6Ii8xMDk1OTg0NTAvNTI1MzgzNzUyLTBiZDQwMTBmLTAyZWMtNGZlNy1iMWMzLTc4OGFjMzkxMmQ2OS5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMjEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTIxM1QyMzMwNDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00ZjJlMjBhMzgzOTY1MmUyM2M5YTU4ZWUxODMwMWE2OTgwNjUxMzExZmYzNjNkNTViNzc3ZmQ3ZDVmMzNmYmJiJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.sjRumjsN073-mbdtYYglPWlwoTbicI76wq4bbxWv7Sw" alt="Demo 2" width="100%" />
      </div>
    </td>
  </tr>

  <tr>
    <td width="50%">
      <h3 align="center">10kg transportation (QP)</h3>
      <div align="center">
        <img src="https://private-user-images.githubusercontent.com/109598450/525366908-43f17544-210b-4458-8ac6-957bcc0f1de7.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjU2Njg5NjQsIm5iZiI6MTc2NTY2ODY2NCwicGF0aCI6Ii8xMDk1OTg0NTAvNTI1MzY2OTA4LTQzZjE3NTQ0LTIxMGItNDQ1OC04YWM2LTk1N2JjYzBmMWRlNy5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMjEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTIxM1QyMzMxMDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zNDBlNjI4YTE1NjVkYzk5MjgyZmQ5NzFjNGFmYzk4YmMwN2NkY2MwOTI1NTNmZjhjMTNiNGU3ODg2YzY3OGMyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.RYfIcTH89SdZAgyMRO_mW8ZPmVHiVN5qXFctkBc5LW4" alt="Demo 3" width="100%" />
      </div>
    </td>
    <td width="50%">
      <h3 align="center">Fast pick-and-place (PID)</h3>
      <div align="center">
        <img src="https://private-user-images.githubusercontent.com/109598450/525385310-1f4f8885-f7f2-4931-975a-7925dd29eb4f.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjU2Njg5ODYsIm5iZiI6MTc2NTY2ODY4NiwicGF0aCI6Ii8xMDk1OTg0NTAvNTI1Mzg1MzEwLTFmNGY4ODg1LWY3ZjItNDkzMS05NzVhLTc5MjVkZDI5ZWI0Zi5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMjEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTIxM1QyMzMxMjZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xNzAyMmVlZDU0ZjlkN2ZkNzAxZGQyNjEzNjY4YTg1MmFlMzExNGU2NTEyM2E4MTFiNTZhYjc1YzVhODE2NjIwJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.QPA127RNLUZy1ANO8-cAlQ6_sm2UdN5My8IgsCzX2Pw" alt="Demo 4" width="100%" />
      </div>
    </td>
  </tr>

  <tr>
    <td width="50%">
      <h3 align="center">Twist movement (PID)</h3>
      <div align="center">
        <img src="https://private-user-images.githubusercontent.com/109598450/525386168-dac76d29-8757-41c6-91dd-9f80ff5dca89.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjU2NjkwMjEsIm5iZiI6MTc2NTY2ODcyMSwicGF0aCI6Ii8xMDk1OTg0NTAvNTI1Mzg2MTY4LWRhYzc2ZDI5LTg3NTctNDFjNi05MWRkLTlmODBmZjVkY2E4OS5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMjEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTIxM1QyMzMyMDFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02Y2E0ZTc2ZDNmYmE2MjUzMDdiMGU1OGZmZWQyMDlmOWIzODA5MDI5YTJhNjI1ZjVlODQ4YTJjNzE4YjllYmE5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.9D4TDCT5_rT278FvS-VBnil4CFlKp1uRmW8QMq-_xus" alt="Demo 5" width="100%" />
      </div>
    </td>
    <td width="50%">
      <h3 align="center">Linear movement (QP)</h3>
      <div align="center">
        <img src="https://private-user-images.githubusercontent.com/109598450/525384976-a5e8ca17-7e9a-4efb-990b-39e19e88834b.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjU2NjkwNDMsIm5iZiI6MTc2NTY2ODc0MywicGF0aCI6Ii8xMDk1OTg0NTAvNTI1Mzg0OTc2LWE1ZThjYTE3LTdlOWEtNGVmYi05OTBiLTM5ZTE5ZTg4ODM0Yi5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMjEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTIxM1QyMzMyMjNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jNzk3NmRkYzY5MGU5MzEwMjU4NDdkM2ZhZjVjNmNlMDIxZGQ4ZTVhYjUyNDhmZmNmNmNmNGE2YzJlYzFhMTU0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.TJf1fTm6QovUtLmMugvHf8nYe6Uuk48zvg2tauSPfRY" alt="Demo 6" width="100%" />
      </div>
    </td>
  </tr>
</table>

## Methodology

## Setup

```bash
pip install -e .
```




