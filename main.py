#!/usr/bin/env python
from manim.constants import DEGREES, ORIGIN, PI, TAU
from manim.utils.color import BLUE, GREEN, RED, YELLOW, color_gradient
from manim.utils.rate_functions import ease_in_expo
import numpy as np
from manim import (
    Arrow,
    Dot3D,
    Rotate,
    ThreeDAxes,
    ThreeDScene,
    TracedPath,
    ValueTracker,
    Write,
    config,
    linear,
    normalize,
    rate_functions,
    smooth,
)
from manim.opengl import (
    OpenGLGroup,
    OpenGLSurface,
    OpenGLSurfaceMesh,
)
from scipy.integrate import solve_ivp

# Define constants
hbar = 1.0  # Reduced Planck's constant
gamma = 2 * PI  # Gyromagnetic ratio

h0 = 0.1
omega = 2 * PI  # Driving / rotating frame frequency


def make_B(beta: float, cutoff: float | None = None):
    def B_x(t: float) -> float:
        if cutoff is None or t < cutoff:
            return h0 * np.cos(omega * t)
        else:
            return 0

    def B_y(t: float) -> float:
        return 0

    def B_z(t: float, omega: float = 0) -> float:
        return 1 + beta + omega / (gamma * hbar)  # Constant in this example

    return B_x, B_y, B_z


def make_schrodinger(beta: float, cutoff: float | None = None):
    # Define time-dependent magnetic field components
    B_x, B_y, B_z = make_B(beta, cutoff)

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


class field_z_rabi(ThreeDScene):
    def construct(self):
        # Settings
        frame_rate = config.frame_rate
        animation_time = 60

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
            resolution=(10, 10),
        )

        time_flow = ValueTracker(0)
        rotation_speed = ValueTracker(0)
        mag_scale = ValueTracker(1)

        self.add(time_flow, axes)

        self.play(Write(bloch_sphere))
        # add the magnetic field vector
        field_vector = Arrow(
            ORIGIN,
            color=RED,
        )

        def magnets(mob):
            try:
                mob.put_start_and_end_on(
                    ORIGIN,
                    axes.c2p(
                        mag_scale.get_value() * B_x(time_flow.get_value()),
                        mag_scale.get_value() * B_y(time_flow.get_value()),
                        mag_scale.get_value()
                        * B_z(time_flow.get_value(), rotation_speed.get_value()),
                    ),
                )
            except:
                pass

        # solve the schrodinger equation

        amount = 10
        betas = np.random.default_rng().normal(0, 0.003, amount)
        betas.sort()
        psi_0 = np.array([np.sqrt(1), np.sqrt(0)], dtype=complex)
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
        colors = color_gradient([GREEN, BLUE], len(betas))

        dots = OpenGLGroup(*(Dot3D(radius=0.05, color=color) for color in colors))
        mean_direction = Arrow(color=YELLOW)

        def update_dots(dots):
            for dot, solution in zip(dots, solutions):
                dot.move_to(axes.c2p(*bloch(solution(time_flow.get_value()))))

        tails = OpenGLGroup(
            *(
                TracedPath(
                    dot.get_center,
                    dissipating_time=12 / (hbar * gamma),
                    stroke_color=color,
                )
                for dot, color in zip(dots, colors)
            )
        )

        self.add(dots)
        self.add(tails)

        def update_mean(mean_dir: Arrow):
            (
                mean_dir.put_start_and_end_on(
                    ORIGIN,
                    axes.c2p(
                        *np.mean(
                            np.array(
                                [
                                    bloch(solution(time_flow.get_value()))
                                    for solution in solutions
                                ]
                            ),
                            axis=0,
                        )
                    ),
                ),
            )

        field_vector.add_updater(magnets, call_updater=True)
        mean_direction.add_updater(update_mean, call_updater=True)
        dots.add_updater(update_dots, call_updater=True)

        self.add(field_vector)
        self.add(mean_direction)

        self.play(time_flow.animate.increment_value(3), run_time=3, rate_func=linear)

        run = 12
        mag_scale.set_value(1 / h0)
        rotation_speed.set_value(-omega)
        self.play(
            time_flow.animate.increment_value(run),
            Rotate(axes, rotation_speed.get_value() * run),
            Rotate(bloch_sphere, rotation_speed.get_value() * run),
            run_time=run,
            rate_func=linear,
        )

        run = 10
        self.play(
            time_flow.animate.increment_value(run),
            Rotate(axes, rotation_speed.get_value() * run),
            Rotate(bloch_sphere, rotation_speed.get_value() * run),
            run_time=run,
            rate_func=linear,
        )

        self.wait(3)


class ft_spectroscopy(ThreeDScene):
    def construct(self):
        # Settings
        frame_rate = config.frame_rate
        animation_time = 60

        cutoff = 1 / (h0 * gamma) * PI

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
            resolution=(10, 10),
        )

        time_flow = ValueTracker(0)
        rotation_speed = ValueTracker(0)
        mag_scale = ValueTracker(1)

        self.add(time_flow, axes)

        self.play(Write(bloch_sphere))
        # add the magnetic field vector
        field_vector = Arrow(
            ORIGIN,
            color=RED,
        )

        B_xc, _, _ = make_B(0, cutoff=cutoff)

        def magnets(mob):
            try:
                mob.put_start_and_end_on(
                    ORIGIN,
                    axes.c2p(
                        mag_scale.get_value() * B_xc(time_flow.get_value()),
                        mag_scale.get_value() * B_y(time_flow.get_value()),
                        mag_scale.get_value()
                        * B_z(time_flow.get_value(), rotation_speed.get_value()),
                    ),
                )
            except:
                pass

        # solve the schrodinger equation

        amount = 10
        betas = np.random.default_rng().normal(0, 0.003, amount)
        betas.sort()
        psi_0 = np.array([np.sqrt(1), np.sqrt(0)], dtype=complex)
        t_span = (0, animation_time)
        t_eval = np.linspace(*t_span, int(animation_time * frame_rate))

        solutions = [
            solve_ivp(
                make_schrodinger(beta, cutoff),
                t_span,
                psi_0,
                t_eval=t_eval,
                method="RK45",
                vectorized=True,
                dense_output=True,
            ).sol
            for beta in betas
        ]
        colors = color_gradient([GREEN, BLUE], len(betas))

        dots = OpenGLGroup(*(Dot3D(radius=0.05, color=color) for color in colors))
        mean_direction = Arrow(color=YELLOW)

        def update_dots(dots):
            for dot, solution in zip(dots, solutions):
                dot.move_to(axes.c2p(*bloch(solution(time_flow.get_value()))))

        tails = OpenGLGroup(
            *(
                TracedPath(
                    dot.get_center,
                    dissipating_time=12 / (hbar * gamma),
                    stroke_color=color,
                )
                for dot, color in zip(dots, colors)
            )
        )

        self.add(dots)
        self.add(tails)

        def update_mean(mean_dir: Arrow):
            (
                mean_dir.put_start_and_end_on(
                    ORIGIN,
                    axes.c2p(
                        *np.mean(
                            np.array(
                                [
                                    bloch(solution(time_flow.get_value()))
                                    for solution in solutions
                                ]
                            ),
                            axis=0,
                        )
                    ),
                ),
            )

        field_vector.add_updater(magnets, call_updater=True)
        mean_direction.add_updater(update_mean, call_updater=True)
        dots.add_updater(update_dots, call_updater=True)

        self.add(field_vector)
        self.add(mean_direction)

        self.play(time_flow.animate.increment_value(3), run_time=3, rate_func=linear)

        run = cutoff - time_flow.get_value()

        mag_scale.set_value(1 / h0)
        rotation_speed.set_value(-omega)
        self.play(
            time_flow.animate.increment_value(run),
            Rotate(axes, rotation_speed.get_value() * run),
            Rotate(bloch_sphere, rotation_speed.get_value() * run),
            run_time=run,
            rate_func=linear,
        )

        run = 20
        self.play(
            time_flow.animate.increment_value(run),
            Rotate(axes, rotation_speed.get_value() * run),
            Rotate(bloch_sphere, rotation_speed.get_value() * run),
            run_time=run,
            rate_func=linear,
        )

        self.wait(3)
