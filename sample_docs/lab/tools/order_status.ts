// Rusty Goose lab — order status messages sent to clients (Telegram bot).
//
// Pickup is only on Wednesdays and Saturdays, 11:00–19:30 (see faq_clients.md).

export const PICKUP_DAYS = [3, 6] as const; // Wednesday, Saturday (Date.getDay())
export const PICKUP_OPEN = { hour: 11, minute: 0 };
export const PICKUP_CLOSE = { hour: 19, minute: 30 };
export const WARRANTY_DAYS = 21;
export const ADDRESS = 'Tomsk, Ryabinovaya st. 12 (yard entrance, mind the goose)';

export type OrderState = 'queued' | 'printing' | 'post-processing' | 'ready' | 'picked-up' | 'failed';

export interface Order {
  id: number;
  client: string;
  state: OrderState;
  printer?: string;
  readyAt?: Date;
}

/** Next pickup window starting from `from` (inclusive). */
export function nextPickup(from: Date): Date {
  const d = new Date(from);
  for (let i = 0; i < 8; i++) {
    const day = d.getDay();
    const isPickupDay = (PICKUP_DAYS as readonly number[]).includes(day);
    const minutes = d.getHours() * 60 + d.getMinutes();
    const open = PICKUP_OPEN.hour * 60 + PICKUP_OPEN.minute;
    const close = PICKUP_CLOSE.hour * 60 + PICKUP_CLOSE.minute;
    if (isPickupDay && minutes < close) {
      if (minutes < open) d.setHours(PICKUP_OPEN.hour, PICKUP_OPEN.minute, 0, 0);
      return d;
    }
    d.setDate(d.getDate() + 1);
    d.setHours(0, 0, 0, 0);
  }
  throw new Error('no pickup window found within a week');
}

export function message(order: Order, now = new Date()): string {
  switch (order.state) {
    case 'queued':
      return `Заказ #${order.id} в очереди. Мы напишем, когда он начнёт печататься.`;
    case 'printing':
      return `Заказ #${order.id} печатается на принтере «${order.printer ?? '?'}».`;
    case 'post-processing':
      return `Заказ #${order.id} напечатан, идёт постобработка (удаление поддержек, шлифовка).`;
    case 'ready': {
      const p = nextPickup(order.readyAt ?? now);
      return (
        `Заказ #${order.id} готов! Забрать можно по средам и субботам с 11:00 до 19:30, ` +
        `ближайшая выдача — ${p.toLocaleDateString('ru-RU')}. Адрес: ул. Рябиновая, 12.`
      );
    }
    case 'picked-up':
      return `Спасибо! Гарантия на заказ #${order.id} действует ${WARRANTY_DAYS} день.`;
    case 'failed':
      return `С заказом #${order.id} возникла проблема при печати. Мы перепечатаем его бесплатно и сообщим новый срок.`;
  }
}
