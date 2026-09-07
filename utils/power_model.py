# -*- coding: utf-8 -*-
"""
utils/power_model.py
Modelos de consumo energético para EdgeServers.
"""


class ServerPowerModel:
    """
    Modelo de consumo de energia de um servidor de borda.

    Placeholder para implementação futura de métricas de energia.
    A referência a esta classe é armazenada no atributo
    EdgeServer.power_model.

    Exemplo de extensão:
        class ServerPowerModel:
            def __init__(self, p_idle=100, p_max=200):
                self.p_idle = p_idle
                self.p_max  = p_max

            def power(self, cpu_util: float) -> float:
                return self.p_idle + (self.p_max - self.p_idle) * cpu_util
    """
    pass
