"""The one resampling a picture goes through between the file and the screen."""
import pytest
import torch
import torch.nn.functional as F

from hdr_blaster.resample import fit_size, resize, spline36


def test_spline36_is_an_interpolating_kernel_three_pixels_wide():
    x = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0, 3.0, 3.5, -7.0], dtype=torch.float64)
    assert spline36(x).tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-12)
    assert float(spline36(torch.tensor(0.5, dtype=torch.float64))) == pytest.approx(0.5987, abs=1e-3)
    assert float(spline36(torch.tensor(1.5, dtype=torch.float64))) < 0.0, "the first lobe is negative: it keeps edges sharp"
    fine = torch.linspace(-3.0, 3.0, 6001, dtype=torch.float64)
    assert float(spline36(fine).sum() * 0.001) == pytest.approx(1.0, abs=1e-3), "it neither adds nor removes light"


def test_a_picture_already_at_the_right_size_is_not_touched(device):
    image = torch.rand(1, 3, 17, 23, device=device)
    assert resize(image, 17, 23) is image


@pytest.mark.parametrize("size", [(9, 14), (30, 41), (16, 64), (64, 16)])
def test_a_flat_picture_stays_flat_and_light_is_kept(device, size):
    flat = torch.full((1, 3, 16, 24), 0.37, device=device)
    out = resize(flat, *size)
    assert out.shape == (1, 3, *size) and out.device == flat.device
    assert torch.allclose(out, torch.full_like(out, 0.37), atol=1e-5)
    picture = torch.rand(2, 3, 48, 64, device=device)
    assert float(resize(picture, 31, 43).mean()) == pytest.approx(float(picture.mean()), abs=0.01)


def test_shrinking_does_not_alias(device):
    """Detail finer than the new grid must be filtered out before the picture is sampled,
    or it comes back as a pattern that is not in the picture."""
    stripes = torch.zeros(1, 1, 8, 400, device=device)
    stripes[..., ::2] = 1.0                                  # one pixel on, one off
    out = resize(stripes, 8, 300)
    inside = out[0, 0, 0, 20:-20]
    # one pixel on, one off is the finest detail a picture can hold: the hardest case. Spline36
    # leaves an eighth of its contrast (a perfect filter would need an endless kernel).
    assert float(inside.std()) < 0.06 and float(stripes.std()) > 0.49, "nearly an even grey"
    assert float(inside.mean()) == pytest.approx(0.5, abs=0.01)


def test_every_pixel_is_taken_where_it_belongs(device):
    """What the window used to do ("area" reduction by a ratio that is not a whole number):
    whole groups of pixels averaged, so that the new pixels are taken up to a third of a
    pixel away from their place, a different amount for each. On a slanted line that is a
    staircase. A ramp shows it plainly: each new pixel must read its own position."""
    ramp = torch.arange(400.0, device=device).expand(1, 1, 8, 400).contiguous()
    place = (torch.arange(300.0, device=device) + 0.5) * 400.0 / 300.0 - 0.5
    naive = F.interpolate(ramp, size=(8, 300), mode="area")[0, 0, 0]
    assert float((naive - place)[20:-20].abs().max()) > 0.3, "the old way: off by a third of a pixel"
    out = resize(ramp, 8, 300)[0, 0, 0]
    assert float((out - place)[20:-20].abs().max()) < 0.02
    larger = resize(ramp, 8, 520)[0, 0, 0]
    place = (torch.arange(520.0, device=device) + 0.5) * 400.0 / 520.0 - 0.5
    assert float((larger - place)[20:-20].abs().max()) < 0.01


def test_a_slanted_edge_stays_a_straight_line(device):
    """Row after row, a slanted edge must advance by the same amount. Where each row
    crosses half-way between dark and light says where the edge is; how far that strays
    from the straight line is the staircase."""
    y, x = torch.meshgrid(torch.arange(240.0, device=device), torch.arange(320.0, device=device), indexing="ij")
    edge = ((x - 0.31 * y - 100.0).clamp(-0.5, 0.5) + 0.5)[None, None]      # an anti-aliased slanted edge

    def wobble(picture, scale):
        rows = picture[0, 0]
        after = (rows > 0.5).float().argmax(1)                               # first pixel past half-way
        low = rows.gather(1, (after - 1)[:, None])[:, 0]
        high = rows.gather(1, after[:, None])[:, 0]
        crossing = (after - 1).float() + (0.5 - low) / (high - low)
        source_row = (torch.arange(float(rows.shape[0]), device=device) + 0.5) / scale - 0.5
        expected = (0.31 * source_row + 100.5) * scale - 0.5
        return float((crossing - expected)[8:-8].abs().max())

    assert wobble(F.interpolate(edge, size=(180, 240), mode="area"), 0.75) > 0.3, "the old way: a staircase"
    out = resize(edge, 180, 240)
    assert wobble(out, 0.75) < 0.15
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, "no halo along the edge"
    assert wobble(resize(edge, 312, 416), 1.3) < 0.15


def test_enlarging_keeps_an_edge_where_it_is(device):
    step = torch.zeros(1, 1, 4, 40, device=device)
    step[..., 20:] = 1.0
    out = resize(step, 4, 60)[0, 0, 0]
    assert float(out[:28].abs().max()) == 0.0 and float((out[32:] - 1.0).abs().max()) == 0.0, "no halo"
    crossing = int((out > 0.5).float().argmax())
    assert crossing == 30


def test_the_picture_fits_its_box_and_keeps_its_shape():
    assert fit_size(2560, 1440, 2560, 1440) == (2560, 1440)
    assert fit_size(1500, 900, 2560, 1440) == (1500, 844)
    assert fit_size(3000, 1500, 2560, 1440) == (2667, 1500), "a larger window enlarges it"
    assert fit_size(1500, 900, 1440, 2560) == (506, 900)
    assert fit_size(10, 10, 2560, 16) == (10, 1), "never less than a pixel"
    # the picture made for a box is recognised by that box: it is then drawn as it is
    for box in ((1500, 900), (1237, 811), (3000, 1500)):
        w, h = fit_size(*box, 2560, 1440)
        assert fit_size(*box, w, h) == (w, h)
