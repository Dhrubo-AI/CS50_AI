import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probs = 1
    children = []
    for person in people:
        gene = (person in one_gene) + 2 * (person in two_genes)
        people[person]["gene"] = gene
        if people[person]["mother"] == None:
            probs *= (PROBS["gene"][gene]*PROBS["trait"][gene][person in have_trait])
        else:
            children.append(person)
    for child in children:
        gene = (child in one_gene) + 2 * (child in two_genes)
        mother = people[child]["mother"]
        ma_gene = people[mother]["gene"]
        inherit_ma = 0.01 * (ma_gene == 0) + 0.5 * 0.99 * (ma_gene == 1) + 0.99 * (ma_gene == 2)
        father = people[child]["father"]
        fa_gene = people[father]["gene"]
        inherit_fa = 0.01 * (fa_gene == 0) + 0.5 * 0.99 * (fa_gene == 1) + 0.99 * (fa_gene == 2)
        if gene == 0:
            probs *= ((1 - inherit_fa) * (1 - inherit_ma))
        elif gene == 1:
            probs *= ((inherit_fa) * (1 - inherit_ma) + (inherit_ma) * (1 - inherit_fa))
        else:
            probs *= (inherit_fa * inherit_ma)
    return probs

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        pgene = (person in one_gene) + 2 * (person in two_genes)
        ptrait = (person in have_trait)
        probabilities[person]["gene"][pgene] += p
        probabilities[person]["trait"][ptrait] += p



def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        gene_sum = 0
        trait_sum = 0
        for gene in probabilities[person]["gene"]:
            gene_sum += probabilities[person]["gene"][gene]
        for trait in probabilities[person]["trait"]:
            trait_sum += probabilities[person]["trait"][trait]
        alpha_trait = 1/trait_sum
        alpha_gene = 1/gene_sum
        for gene in probabilities[person]["gene"]:
            probabilities[person]["gene"][gene] *= alpha_gene
        for trait in probabilities[person]["trait"]:
            probabilities[person]["trait"][trait] *= alpha_trait


if __name__ == "__main__":
    main()
