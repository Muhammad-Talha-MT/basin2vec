"""End-to-end checks against a running static server; writes only ignored QA files."""
import csv
import io
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = os.environ.get("BASIN_ATLAS_URL", "http://127.0.0.1:4173")
ARTIFACTS = Path(__file__).resolve().parents[1] / "qa-artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def ready(page):
    page.wait_for_function("window.basinAtlas && !window.basinAtlas.getState().busy && window.basinAtlas.getPoints()", timeout=90000)


def lookup(page, basin="04101500"):
    page.locator("#search").fill(basin)
    page.locator("#search").press("Enter")
    ready(page)
    assert page.evaluate("basinAtlas.getBasins()[basinAtlas.getState().selected].id") == basin


def ranking(page, expected, value):
    result = page.evaluate("""() => {
      const a=basinAtlas, scores=a.getScores(), index=a.getState().selected;
      return Array.from(scores,(score,i)=>({score,i})).filter(x=>x.i!==index)
        .sort((a,b)=>b.score-a.score).slice(0,5).map(x=>({id:a.getBasins()[x.i].id,score:x.score}));
    }""")
    assert [r['id'] for r in result] == expected, result
    assert abs(result[0]['score'] - value) < 1e-6


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    context = browser.new_context(viewport={"width":1440,"height":1000}, device_scale_factor=1, accept_downloads=True)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(BASE, wait_until="networkidle")
    ready(page)
    assert page.locator(".examples button").count() == 3
    page.screenshot(path=str(ARTIFACTS / "desktop-overview.png"), full_page=True)
    lookup(page)
    ranking(page, ['04183000','03282290','01154500','05552500','03284500'], .95626390)
    assert page.locator(".neighbor").count() == 10
    page.locator("#reset-map").click()
    page.screenshot(path=str(ARTIFACTS / "desktop-similarity.png"), full_page=True)
    with page.expect_download() as event:
        page.locator("#export").click()
    rows = list(csv.DictReader(io.StringIO(Path(event.value.path()).read_text())))
    assert len(rows) == 9066 and rows[0]['usgs_id'] == '04183000'
    assert all(row['usgs_id'] != '04101500' for row in rows)
    page.locator("#exclude-overlap").check()
    page.locator("#distance").select_option('500')
    with page.expect_download() as event:
        page.locator("#export").click()
    rows = list(csv.DictReader(io.StringIO(Path(event.value.path()).read_text())))
    overlaps = page.evaluate("basinAtlas.getBasins()[basinAtlas.getState().selected].overlaps.map(([i])=>basinAtlas.getBasins()[i].id)")
    assert rows and all(float(row['centroid_distance_km']) >= 499.999 and row['usgs_id'] not in overlaps for row in rows)
    page.locator("#distance").select_option('0')
    page.locator("#exclude-overlap").uncheck()
    page.locator('[data-tab="signature"]').click()
    page.wait_for_selector('#trend svg circle')
    assert page.locator('.signature-cell').count() == 64
    assert page.locator('#trend svg circle').count() == 40
    with page.expect_download() as event:
        page.locator('#download-vector').click()
    vector = json.loads(Path(event.value.path()).read_text())
    assert len(vector['values']) == 64 and vector['usgs_id'] == '04101500'
    assert abs(sum(x*x for x in vector['values']) - 1) < 1e-5
    page.locator('#year').evaluate("el => {el.value='2024';el.dispatchEvent(new Event('input',{bubbles:true}));}")
    ready(page)
    ranking(page, ['03282290','03282000','03282060','01154500','04113000'], .95343810)
    page.locator('#space').select_option('hydrologic_repr')
    ready(page)
    ranking(page, ['03282290','03282000','05430500','07297910','07232500'], .99417150)
    page.locator('#mean').click()
    ready(page)
    ranking(page, ['03282290','03282060','01029500','11527000','08136000'], .98725289)
    page.locator('#year').evaluate("el => {for(const y of [1987,2000,2024]){el.value=y;el.dispatchEvent(new Event('input',{bubbles:true}));}}")
    ready(page)
    assert page.evaluate("basinAtlas.getState().period") == '2024'
    ranking(page, ['03282290','03282000','05430500','07297910','07232500'], .99417150)
    page.locator('[data-mode="change"]').click()
    assert '1985 → 2024' in page.locator('#legend').inner_text()
    page.reload(wait_until='networkidle')
    ready(page)
    assert page.evaluate("basinAtlas.getState().spaceKey") == 'hydrologic_repr'
    assert page.evaluate("basinAtlas.getState().period") == '2024'
    assert page.evaluate("basinAtlas.getState().mode") == 'change'
    ranking(page, ['03282290','03282000','05430500','07297910','07232500'], .99417150)

    # Find a real nested location away from floating controls, then physically click it.
    location = page.evaluate("""() => {
      const a=basinAtlas, size=a.map.getSize();
      for(let lat=36;lat<44;lat+=.25)for(let lon=-100;lon<-77;lon+=.25){
        const pt=a.map.latLngToContainerPoint([lat,lon]);
        if(pt.x<330||pt.y<80||pt.y>size.y-175)continue;
        const hits=a.hitsAt([lat,lon]);
        if(hits.length>=3)return {lat,lon,x:pt.x,y:pt.y,hits};
      }
    }""")
    assert location, 'No test overlap location found'
    rect = page.locator('#map').bounding_box()
    page.mouse.click(rect['x']+location['x'],rect['y']+location['y'])
    page.wait_for_selector('#overlap-picker:not([hidden])')
    assert page.locator('.pick').count() == len(location['hits'])
    assert page.evaluate("indices=>indices.map(i=>basinAtlas.getBasins()[i].area)",location['hits']) == sorted(page.evaluate("indices=>indices.map(i=>basinAtlas.getBasins()[i].area)",location['hits']))
    page.screenshot(path=str(ARTIFACTS / 'desktop-overlap.png'), full_page=True)
    page.locator('.pick').nth(1).click()
    ready(page)
    assert page.evaluate('basinAtlas.getState().selected') == location['hits'][1]
    page.locator('[data-tab="overlaps"]').click()
    assert page.locator('.overlap-item').count() == page.evaluate('basinAtlas.getBasins()[basinAtlas.getState().selected].overlaps.length')
    page.locator('#gauges').check()
    page.locator('#gauges').uncheck()
    page.locator('#outlines').uncheck()
    page.locator('#outlines').check()
    page.locator('#search').fill('no-such-basin-123')
    assert 'No matching basins' in page.locator('#search-results').inner_text()
    page.locator('#search').press('Escape')
    page.locator('#about').click()
    assert page.locator('#about-dialog').is_visible()
    page.keyboard.press('Escape')
    assert not page.locator('#about-dialog').is_visible()
    assert not errors, errors

    mobile = browser.new_page(viewport={'width':390,'height':844})
    mobile.on('pageerror',lambda error:errors.append(str(error)))
    mobile.goto(BASE,wait_until='networkidle')
    ready(mobile)
    assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    lookup(mobile)
    mobile.locator('#reset-map').click()
    mobile.screenshot(path=str(ARTIFACTS / 'mobile.png'),full_page=True)
    mobile.locator('[data-tab="signature"]').click()
    mobile.wait_for_selector('#trend svg circle')
    assert mobile.locator('.signature-cell').count() == 64
    assert not errors, errors

    failure = context.new_page()
    failure.route('**/data/embeddings/avg.f32.gz',lambda route:route.fulfill(status=503,body='unavailable'))
    failure.goto(BASE)
    failure.wait_for_selector('#retry',timeout=90000)
    assert '503' in failure.locator('#loading').inner_text()
    assert failure.locator('#export').is_disabled()
    browser.close()
    print('PASS: real-data rankings, search, overlap hit chooser, annual/space switches, race handling, filters, CSV/JSON, trends, deep links, mobile, and data-failure recovery. No page errors.')
