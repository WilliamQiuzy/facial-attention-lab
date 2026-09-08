"""UX 1–7 acceptance: visible actions, compact navigation, and retained media.

Uses synthetic bytes only. Recording/model/PDF loops remain in live_full_loop.py.
"""
import argparse
import hashlib
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright
from journey_edge_acceptance import _route_ready


def useful_button(button, height):
    box = button.bounding_box()
    assert box and box['y'] >= 0 and box['y'] + box['height'] <= height + 1, box
    assert box['height'] >= 48, box
    assert float(button.evaluate('e => parseFloat(getComputedStyle(e).fontSize)')) >= 16


def check_layout(browser, name, base, width, height, output):
    page = browser.new_page(viewport={'width': width, 'height': height}, reduced_motion='reduce')
    _route_ready(page)
    page.goto(base)
    page.get_by_role('radio', name='No — standard assessment', exact=False).check()
    expect(page.get_by_text('Movement 1 of 7', exact=True)).to_be_visible()
    page.get_by_role('button', name='Continue to camera setup', exact=True).click()
    enable = page.get_by_role('button', name='Enable camera', exact=True)
    expect(enable).to_have_count(1)
    useful_button(enable, height)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
    if height <= 560:
        rail = page.locator('.workflow-section')
        assert rail.bounding_box()['height'] <= 80
        compact = page.locator('.workflow-compact-menu summary')
        expect(compact).to_be_visible()
        compact.click()
        page.locator('.workflow-compact-menu').get_by_role('button', name='1. Prepare').click()
        expect(page.get_by_role('heading', name='Before recording', exact=True)).to_be_visible()
        page.get_by_role('button', name='Continue to camera setup', exact=True).click()
    if name == 'chromium':
        page.screenshot(path=output / f'setup-{width}x{height}.png')
    page.close()


def check_retention(browser, base):
    page = browser.new_page(viewport={'width': 1366, 'height': 768}, reduced_motion='reduce')
    _route_ready(page)
    page.goto(base)
    page.get_by_role('radio', name='No — standard assessment', exact=False).check()
    page.get_by_role('button', name='Continue to camera setup').click()
    page.get_by_role('tab', name='Upload from LifeLink').click()
    expect(page.get_by_text('You need the video and its matching FACES action timeline (.json).', exact=False)).to_be_visible()
    page.get_by_label('Choose LifeLink Face video').set_input_files({
        'name': 'synthetic-session.webm', 'mimeType': 'video/webm', 'buffer': b'synthetic-video',
    })
    ids = ['neutral_repose', 'eyebrow_raise', 'gentle_eye_closure', 'tight_eye_squeeze', 'relaxed_smile', 'lip_pucker', 'lower_teeth_show']
    timeline = {
        'schema_version': 'faces-action-timeline/v1', 'script_version': 'faces-script/24-004956-v1',
        'recording_sha256': hashlib.sha256(b'synthetic-video').hexdigest(),
        'timing_source': 'capture_event_log', 'recording_duration_ms': 28000,
        'actions': [{'action': action, 'status': 'completed', 'prompt_start_ms': i*4000,
                     'hold_start_ms': i*4000+500, 'hold_end_ms': i*4000+3500,
                     'completion_ms': i*4000+3750} for i, action in enumerate(ids)],
    }
    page.get_by_label('Choose FACES action timeline').set_input_files({
        'name': 'synthetic-session.timeline.json', 'mimeType': 'application/json',
        'buffer': json.dumps(timeline).encode(),
    })
    expect(page.get_by_role('heading', name='Review the recording and run analysis')).to_be_visible()
    page.get_by_role('button', name='Return to Prepare', exact=True).click()
    page.get_by_role('button', name='Continue to camera setup').click()
    expect(page.get_by_text('synthetic-session.webm', exact=True)).to_be_visible()
    # Browser Back changes the stage, not the recording's lifetime.
    page.go_back()
    expect(page.get_by_role('heading', name='Before recording', exact=True)).to_be_visible()
    page.get_by_role('button', name='Return to Analyze', exact=True).click()
    page.get_by_role('button', name='Clear recording and start over').click()
    dialog = page.get_by_role('dialog')
    expect(dialog.get_by_role('button', name='Keep current recording')).to_be_focused()
    page.keyboard.press('Escape')
    expect(dialog).to_have_count(0)
    expect(page.get_by_text('synthetic-session.webm', exact=True)).to_be_visible()
    page.get_by_role('button', name='Return to live camera').click()
    expect(page.get_by_role('dialog')).to_be_visible()
    page.get_by_role('button', name='Discard and continue').click()
    expect(page.get_by_role('button', name='Enable camera', exact=True)).to_be_visible()
    page.get_by_role('tab', name='Upload from LifeLink').click()
    expect(page.get_by_text('synthetic-session.webm', exact=True)).to_have_count(0)
    page.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8081')
    parser.add_argument('--output', type=Path, default=Path('/tmp/faces-ux-improvements'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as pw:
        for name in ('chromium', 'firefox', 'webkit'):
            browser = getattr(pw, name).launch()
            for width, height in ((320, 800), (390, 844), (768, 900), (1366, 768), (844, 390), (1280, 480)):
                check_layout(browser, name, args.base_url, width, height, args.output)
                results.append(f'{name}:{width}x{height}:visible-action-and-navigation')
            check_retention(browser, args.base_url)
            results.append(f'{name}:media-protection-and-browser-back')
            browser.close()
    (args.output / 'results.json').write_text(json.dumps({'passed': results}, indent=2))
    print(f'PASS {len(results)} UX cases across Chromium, Firefox and WebKit')


if __name__ == '__main__':
    main()
