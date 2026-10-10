# SPDX-License-Identifier: MPL-2.0
"""Prepare a pinned external HAMS source candidate; never build or execute it."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


class ProducerRecipeError(ValueError):
    """Unknown source, unsafe destination or changed recipe."""


def prepare_candidate(source, output, *, gnu_compatible=False, preserve_frequency_limits=False, consistent_gravity=False):
    source=Path(source).resolve();output=Path(output).resolve()
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ProducerRecipeError('Source and output must be separate non-nested paths')
    if output.exists():raise FileExistsError(output)
    if not isinstance(gnu_compatible,bool):raise ProducerRecipeError('Explicit GNU compatibility boolean required')
    if not isinstance(preserve_frequency_limits,bool):raise ProducerRecipeError('Explicit limit preservation boolean required')
    if preserve_frequency_limits and not gnu_compatible:raise ProducerRecipeError('Limit reorder replacement has GNU-only evidence; explicit GNU option required')
    if not isinstance(consistent_gravity,bool):raise ProducerRecipeError('Explicit gravity consistency boolean required')
    if consistent_gravity and not (gnu_compatible and preserve_frequency_limits):raise ProducerRecipeError('Gravity correction requires the evidenced GNU limit-preserving source profile')
    recipe=json.loads(Path(__file__).with_name('hams_phase_source_recipe.json').read_text())
    if recipe.get('format')!='anyhydro-hams-phase-source-recipe-v1':raise ProducerRecipeError('Unsupported recipe')
    inventory={str(p.relative_to(source)).replace('\\','/') for p in (source/'src').rglob('*') if p.is_file() and p.suffix.lower()=='.f90'}
    if inventory!=set(recipe['source_sha256']):raise ProducerRecipeError('Compiled Fortran source inventory differs from pin')
    for name,expected in recipe['source_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise ProducerRecipeError(f'Source identity differs: {name}')
    # Check all preconditions before writing. Never modify the supplied tree.
    replacements=[(name,recipe['phase_before'].encode('ascii'),recipe['phase_after'].encode('ascii')) for name in recipe['phase_writers']]
    if gnu_compatible:
        port=recipe['gnu_compatibility'];replacements.append((port['path'],port['before'].encode('ascii'),port['after'].encode('ascii')))
    if preserve_frequency_limits:
        option=recipe['limit_reorder'];name=option['path'];raw=(source/name).read_bytes()
        start=raw.index(option['start'].encode('ascii'));end=raw.index(option['end'].encode('ascii'),start)+len(option['end'])
        old=raw[start:end]
        asset=Path(__file__).with_name(option['replacement_file']);new=asset.read_bytes().rstrip(b'\r\n')
        if hashlib.sha256(old).hexdigest()!=option['before_sha256'] or hashlib.sha256(asset.read_bytes()).hexdigest()!=option['replacement_sha256']:
            raise ProducerRecipeError('Limit reorder source/replacement identity differs')
        replacements.append((name,old,new))
    if consistent_gravity:
        option=recipe['gravity_consistency'];name=option['path'];raw=(source/name).read_bytes()
        start=raw.index(option['start'].encode('ascii'));end=raw.index(option['end'].encode('ascii'),start)+len(option['end'])
        old=raw[start:end];new=old
        if hashlib.sha256(old).hexdigest()!=option['before_sha256']:raise ProducerRecipeError('Dispersion source identity differs')
        for edit in option['edits']:
            before=edit['before'].encode('ascii');after=edit['after'].encode('ascii')
            if new.count(before)!=1:raise ProducerRecipeError('Dispersion correction must match exactly once')
            new=new.replace(before,after)
        replacements.append((name,old,new))
    for name,old,new in replacements:
        if (source/name).read_bytes().count(old)!=1:raise ProducerRecipeError('Replacement must match exactly once: '+name)
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ProducerRecipeError('Symlinked source trees are unsupported')
    shutil.copytree(source,output,ignore=shutil.ignore_patterns('.git','__pycache__'))
    for name,old,new in replacements:
        p=output/name;p.write_bytes(p.read_bytes().replace(old,new))
    hashes={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in recipe['source_sha256']}
    changed=sorted(name for name in hashes if hashes[name]!=recipe['source_sha256'][name])
    if changed!=sorted({name for name,_,_ in replacements}):raise ProducerRecipeError('Unexpected source mutation')
    manifest=dict(format='anyhydro-hams-phase-source-candidate-v1',upstream_commit=recipe['upstream_commit'],upstream_license=recipe['upstream_license'],recipe_sha256=hashlib.sha256(Path(__file__).with_name('hams_phase_source_recipe.json').read_bytes()).hexdigest(),source_sha256=recipe['source_sha256'],candidate_sha256=hashes,changed_files=changed,gnu_compatible=gnu_compatible,compiler_source_order=recipe['compiler_source_order'],solver_executed=False,qualified=False,canonical_convention_admitted=False)
    if preserve_frequency_limits:
        manifest.update(format='anyhydro-hams-limit-source-candidate-v2',preserve_frequency_limits=True,limit_reorder_replacement_sha256=recipe['limit_reorder']['replacement_sha256'],limit_damping='absent; four-field rows preserved')
    if consistent_gravity:
        manifest.update(format='anyhydro-hams-gravity-source-candidate-v3',consistent_gravity=True,gravity=recipe['gravity_consistency']['gravity'],dispersion_source='Const_mod.G',known_prior_limitation='Local dispersion gravity9.807 conflicted with global9.80665; prior identities preserved, no output renormalization')
    (output/'phase-candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,help='Pristine exact pinned upstream source directory')
    parser.add_argument('--output',required=True,help='New external candidate directory')
    parser.add_argument('--gnu-compatible',action='store_true',help='Apply shared directory-INQUIRE portability edit for GNU builds')
    parser.add_argument('--preserve-frequency-limits',action='store_true',help='Opt-in GNU fixed-width limit-row reorder repair; no backend admission')
    parser.add_argument('--consistent-gravity',action='store_true',help='Opt-in pinned DISPERSION use of existing global gravity; requires GNU and limit-preservation flags')
    args=parser.parse_args()
    try:result=prepare_candidate(args.source,args.output,gnu_compatible=args.gnu_compatible,preserve_frequency_limits=args.preserve_frequency_limits,consistent_gravity=args.consistent_gravity)
    except (OSError,ValueError,KeyError,TypeError) as exc:parser.exit(2,f'Source preparation refused: {exc}\n')
    print(json.dumps(dict(changed_files=result['changed_files'],qualified=False,solver_executed=False)))


if __name__=='__main__':main()
