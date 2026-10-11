# Обзор материнской платы
[Фотографии материнской платы v1.5](https://drive.google.com/drive/folders/178NaIAVjkrnVmxGbYWasyJjooYn8aHQS?usp=sharing)

Материнская плата D6 «во многом вдохновлена»
[материнской платой Ultimaker 2](https://github.com/Ultimaker/Ultimaker2/tree/master/1091_Main_board_v2.1.1_(x1))
Она построена на микроконтроллере ATMega2560. В основном это схема на 24 вольта с несколькими выводами на 5 вольт.

![](/images/reprap/wanhao/duplicator6/motherboard/motherboard.png)

### Интересные выводы
- J9 — слышал, что его используют для датчика Z, [нужно больше информации]
- J19 — сюда для удобства подключения добавили 2-контактный разъём JST «мама». Очень рекомендуется!
- J16 + J17 — кто-нибудь знает, для чего они (Safety 1+2) могли бы использоваться?
- Судя по схемам платы Ultimaker2, следующее МОЖЕТ БЫТЬ верным и полезным:
  - J9 — Analog — на контакте 2 (средний) должно быть 5 В
  - J16 — Safety 1 — на контакте 1 должно быть 5 В, контакт 2 общий с контактом 1 разъёма J17
  - D7 — зелёный светодиод питания
  - D3 — жёлтый светодиод RX — ?
  - D4 — жёлтый светодиод TX — ?
## Драйверы шаговых моторов
[Лучшее, что мог предложить 2015 год: Allegro A4988ET для осей X, Y, Z и экструдера (PDF)](https://www.allegromicro.com/~/media/Files/Datasheets/A4988-Datasheet.ashx)
## Реле
Типичное слабое место этой версии материнской платы — реле цепи 5 В (большой синий кубик). Оно может вызывать множество проблем, когда принтер включается «как бы наполовину»: реле отключает цепь 5 В, от которой питается передняя панель — ЖК-дисплей и ручка.

Если коротко: обойдите реле, замкнув его одножильным проводом, — это решает МНОГИЕ проблемы. Ниже моя ужасная пайка, которая прекрасно работает!

![](/images/reprap/wanhao/duplicator6/motherboard/shorting_relay.jpg)

Подробнее о проблемах с реле:

- [Официальное видео Wanhao об обходе реле на материнской плате](https://www.youtube.com/watch?v=0HC_GcVOXU0)
- [Официальный PDF Wanhao об обходе реле на материнской плате](http://www.wanhao3dprinter.com/xiazai/D6_Bypass_relay.pdf)
- [Исследование печально известного реле Monoprice Maker Ultimate (Wanhao Duplicator 6)](https://newscrewdriver.com/2017/10/24/investigating-the-infamous-relay-of-monoprice-maker-ultimate-wanhao-duplicator-6/)
## Шлейф экструдера
![](/images/reprap/wanhao/duplicator6/major_components/mobo_extruder_pinout.png)

Контакты E-1A, 1B, 2A, 2B используются шаговым мотором экструдера.

[Ссылка на мою таблицу Google с распиновкой](https://docs.google.com/spreadsheets/d/1XX6I6iNZCkVIueA0zD95mo8jjTG3_VZ3gDfGH6G9hfs/edit?usp=sharing)

## MOSFET-транзисторы
[TODO: перевести фото в текст](https://drive.google.com/drive/folders/1GbxoP8aaD2Qk7poxObGDBJZ8CKOqrU9_?usp=sharing)

## Микросхема T1
Микросхема T1 рядом с разъёмом шлейфа экструдера на материнской плате управляет обоими вентиляторами экструдера. Если на один или оба вентилятора не приходит питание, возможно, эту микросхему нужно заменить или обойти. Проверьте, нет ли в ней короткого замыкания; обычно они полностью уходят в обрыв, так что можно напаять другую прямо поверх. Вместо T-1 подойдёт практически любой NPN-транзистор общего назначения.

Оригинальная микросхема — NPN-транзистор BC817

![](/images/reprap/wanhao/duplicator6/motherboard/before.jpg)

Её можно заменить похожей микросхемой или распространённым транзистором 2N2222.

![](/images/reprap/wanhao/duplicator6/motherboard/after_from_ralph_burki_w_labels.jpg)
