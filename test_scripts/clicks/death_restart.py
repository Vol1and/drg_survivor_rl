import time
from env.ui.ui_controller import UIController


def main():
    ui = UIController()

    print("⚠️ Наведи игру на экран смерти.")
    print("Клик произойдёт через 3 секунды...")
    time.sleep(3)

    print("➡️ Нажимаю кнопку смерти")
    ui.handle_death_restart()

    print("✅ Клик отправлен")


if __name__ == "__main__":
    main()
