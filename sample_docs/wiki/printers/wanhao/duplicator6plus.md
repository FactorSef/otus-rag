# Wanhao Duplicator 6+
Похоже, нужные соединения уже выведены на шлейф (нижние 4 контакта разъёма шлейфа). Z-MIN соответствует выводу 29, а BL-SIG — выводу 9. Это относительно платы Arduino Mega 2560. Мне кажется, они также планировали датчик окончания филамента. FLA-DET на плате соответствует выводу 12. На J10 CTL — вывод 10, BOTTON — вывод 11. На J11 AC-DET — вывод A9. TX и RX идут на TXD2 (вывод 16) и RXD2 (вывод 17)
выводы на J10 (CTL — вывод 10, BOTTON — вывод 11)

Я отредактировал pins_ULTIMAIN_2.h, указав найденные выводы для Z-MIN и BL-SIG, и, похоже, всё работает нормально. Хорошая новость в том, что большинство выводов оставили такими же, как на старом D6, — у меня только экструдер крутился не в ту сторону.

![](/images/reprap/wanhao/duplicator6/duplicator6plus/florian_gabriel_plus_mobo_flop.jpg)

![](/images/reprap/wanhao/duplicator6/duplicator6plus/hans_coetzee_plus_mobo_back.jpg)
