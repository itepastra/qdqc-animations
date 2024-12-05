#!/usr/bin/env python
from manim import *
from manim.opengl import (
    OpenGLGroup,
    OpenGLSurface,
    OpenGLSurfaceMesh,
    OpenGLVGroup,
    OpenGLVMobject,
)
import numpy as np
from scipy.integrate import solve_ivp


# Define constants
hbar = 1.0  # Reduced Planck's constant
gamma = 8.0  # Gyromagnetic ratio

h0 = 0.1
omega = 8


def make_B(beta: float):
    def B_x(t: float) -> float:
        return h0 * np.cos(omega * t)

    def B_y(t: float) -> float:
        return 0

    def B_z(t: float) -> float:
        return 1 + beta  # Constant in this example

    return B_x, B_y, B_z


def make_schrodinger(beta: float):
    # Define time-dependent magnetic field components
    B_x, B_y, B_z = make_B(beta)

    # Hamiltonian as a function of time
    def hamiltonian(t: float):
        return (
            0.5
            * gamma
            * np.array(
                [[B_z(t), B_x(t) - 1j * B_y(t)], [B_x(t) + 1j * B_y(t), -B_z(t)]]
            )
        )

    # Schrödinger equation with time-dependent Hamiltonian
    def schrodinger(t: float, psi):
        H = hamiltonian(t)
        return -1j / hbar * H @ psi

    return schrodinger


B_x, B_y, B_z = make_B(0)


def bloch(psi):
    alpha, beta = psi[0], psi[1]
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha) ** 2 - np.abs(beta) ** 2
    return x, y, z


class field_z(ThreeDScene):
    def construct(self):
        # Settings
        frame_rate = config.frame_rate
        animation_time = 25

        # set the scene
        self.set_camera_orientation(phi=5 * PI / 12, theta=PI / 6)

        axes = ThreeDAxes(
            x_range=[-1.5, 1.5, 0.5], y_range=[-1.5, 1.5, 0.5], z_range=[-1.5, 1.5, 0.5]
        )

        bloch_sphere = OpenGLSurfaceMesh(
            OpenGLSurface(
                lambda u, v: np.array(
                    axes.c2p(*[np.cos(u) * np.cos(v), np.cos(u) * np.sin(v), np.sin(u)])
                ),
                v_range=[0, TAU],
                u_range=[-PI / 2, PI / 2],
                resolution=(16, 16),
            ),
            color="#ff00ff",
            resolution=(9, 9),
        )

        time_text = ValueTracker(0)

        self.add(time_text, axes)

        self.play(Write(bloch_sphere))
        # add the magnetic field vector
        field_vector = Arrow(
            ORIGIN,
            color=RED,
        )

        def magnets(mob):
            mob.put_start_and_end_on(
                ORIGIN,
                axes.c2p(
                    B_x(time_text.get_value()),
                    B_y(time_text.get_value()),
                    B_z(time_text.get_value()),
                ),
            )

        # solve the schrodinger equation

        amount = 25
        betas = np.random.default_rng().normal(0, 0.1, amount)
        betas.sort()
        psi_0 = np.array([np.sqrt(0.7), np.sqrt(0.3)], dtype=complex)
        t_span = (0, animation_time)
        t_eval = np.linspace(*t_span, int(animation_time * frame_rate))

        solutions = [
            solve_ivp(
                make_schrodinger(beta),
                t_span,
                psi_0,
                t_eval=t_eval,
                method="RK45",
                vectorized=True,
                dense_output=True,
            ).sol
            for beta in betas
        ]
        colors = color_gradient([GREEN, BLUE], amount)

        dots = OpenGLGroup(*(Dot3D(radius=0.05, color=color) for color in colors))
        mean_direction = Arrow(color=YELLOW)

        def update_dots(dots):
            for dot, solution in zip(dots, solutions):
                dot.move_to(axes.c2p(*bloch(solution(time_text.get_value()))))

        tails = OpenGLGroup(
            *(
                TracedPath(
                    dot.get_center, dissipating_time=1 / gamma, stroke_color=color
                )
                for dot, color in zip(dots, colors)
            )
        )

        self.add(dots)
        self.add(tails)

        def update_mean(mean_dir):
            mean_dir.put_start_and_end_on(
                ORIGIN, np.mean(np.array([dot.get_center() for dot in dots]), axis=0)
            )

        field_vector.add_updater(magnets, call_updater=True)
        mean_direction.add_updater(update_mean, call_updater=True)
        dots.add_updater(update_dots, call_updater=True)

        self.add(field_vector)
        self.add(mean_direction)

        self.play(time_text.animate.increment_value(3), run_time=3, rate_func=linear)
        self.begin_ambient_camera_rotation(omega )
        self.play(time_text.animate.increment_value(7), run_time=7, rate_func=linear)
