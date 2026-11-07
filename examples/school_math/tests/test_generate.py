import random
from dataclasses import dataclass

from examples.school_math.generate_dataset import stable_expand_entities


@dataclass(eq=False)
class ExampleEntity:
    _description: str
    handle: str
    hash_value: int
    child_description: str = ""
    expression_used: bool = False

    def __hash__(self) -> int:
        return self.hash_value

    @property
    def description(self) -> str:
        return self._description


@dataclass
class ExampleContext:
    child_entities: list[ExampleEntity]


def test_composed_statement_order_does_not_depend_on_entity_identity_hash() -> None:
    first = ExampleContext(
        [
            ExampleEntity("Let a = 3.", "a", 1),
            ExampleEntity("Let b = 2.", "b", 2),
        ]
    )
    second = ExampleContext(
        [
            ExampleEntity("Let a = 3.", "a", 2),
            ExampleEntity("Let b = 2.", "b", 1),
        ]
    )
    random.seed(123)
    expected = stable_expand_entities(
        first, ExampleEntity, target=first.child_entities[0]
    )
    random.seed(123)
    actual = stable_expand_entities(
        second, ExampleEntity, target=second.child_entities[0]
    )
    assert expected == actual
    assert actual[0].count("Let a = 3.") == 1
    assert actual[1] == {"target": "a"}


def test_composed_adapter_keeps_dependencies_and_excludes_expression_only_description() -> (
    None
):
    entity = ExampleEntity(
        "unused expression handle",
        "x",
        1,
        child_description="Let a = 3.",
        expression_used=True,
    )
    prefix, arguments = stable_expand_entities(
        ExampleContext([entity]), ExampleEntity, number=4
    )
    assert prefix == "Let a = 3."
    assert arguments == {"number": 4}
