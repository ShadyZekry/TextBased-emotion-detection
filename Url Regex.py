import re

inputString = """add1 http://mit.edu.com abc
add2 https://facebook.jp.com.2. abc
add3 www.google.be. uvw
add4 https://www.google.be. 123
add5 www.website.gov.us test2
Hey bob on www.test.com. 
another test with ipv4 http://192.168.1.1/test.jpg. toto2
website with different port number www.test.com:8080/test.jpg not port 80
www.website.gov.us/login.html
test with ipv4 (192.168.1.1/test.jpg).
search at google.co.jp/maps.
test with ipv6 2001:0db8:0000:85a3:0000:0000:ac1f:8001/test.jpg."""

regex = r"\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b"

matches = re.findall(regex, inputString)
print(matches)

# \b ----> is used for word boundary to delimit the URL and the rest of the text
# (?:https?://)? ----> to match http:// or https// if present
# (?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6}) ----> to match standard url (that might start with www. (lets call it standerd url)
# (?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?) ---->  to match standard Ipv4 (lets call it IPv4)

# """
# to match the IPv6 URLs:
# (?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|
# (?:[0-9a-fA-F]{1,4}:){1,7}:|
# (?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|
# (?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|
# (?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|
# (?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|
# (?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|
# [0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|
# fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|
# ::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|
# (?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|
# (?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|
# (?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])) (lets call it IPv6)
# """
# to match the port part (lets call it PORT) if present ----> (?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])
# to match the (?:/[\w\.-]*)*/?) target object part of the url (html file, jpg,...) (lets call it RESSOURCE_PATH)