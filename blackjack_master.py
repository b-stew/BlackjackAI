import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

# Define actions
ACTIONS = ["hit", "stand", "double", "split"]  
EPSILON = 1.0  # Exploration rate
ALPHA = 0.001  # Initial learning rate
GAMMA = 0.999  # Discount factor
EPISODES = 200000  # Training episodes
EVAL_INTERVAL = 1000  # Plot eval interval
Q_TABLE_FILE = 'q_table.pkl'

# Define card/deck
CARD_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K', 'A']
DECK = CARD_VALUES * 8

# Caculates the value of face cards
def card_value(card):
    if card in ['J', 'Q', 'K']:
        return 10
    elif card == 'A':
        return 11
    else:
        return card

# Checks whether hand can be split
def can_split_hand(player_hand):
    return len(player_hand) == 2 and card_value(player_hand[0]) == card_value(player_hand[1])


class BlackjackEnv():
    # Initializes environment
    def __init__(self):
        self.deck = self.create_deck()
        self.running_count = 0  
        self.current_hand = 0
        self.player_hands = []  
        self.dealer_hand = []
    
    #Creates and shuffles new deck
    def create_deck(self):
        deck = DECK[:]
        random.shuffle(deck)
        return deck
    
    # Checks whetehr deck needs to be shuffled due to running out of cards
    def should_shuffle(self):
        return len(self.deck) < 10

    # Deals a card to whatever player is calling function
    # Also implements the running count
    def deal_card(self):
        if self.should_shuffle():
            self.deck = self.create_deck()
            self.running_count = 0
        card = self.deck.pop()
        if card in [2, 3, 4, 5, 6]:
            self.running_count += 1
        elif card in [10, 'J', 'Q', 'K', 'A']:
            self.running_count -= 1
        return card

    #Checks the value of a hand and handles aces
    def hand_value(self, hand):
        value = sum(card_value(card) for card in hand)
        num_aces = hand.count('A')
        while value > 21 and num_aces:
            value -= 10
            num_aces -= 1
        return value
    
    # Resets table and deals out new cards
    def reset_table(self):
        self.player_hands = [[]] 
        self.dealer_hand = []   
        self.player_hands[0].append(self.deal_card())
        self.dealer_hand.append(self.deal_card())
        self.player_hands[0].append(self.deal_card())
        self.dealer_hand.append(self.deal_card())
        self.current_hand = 0
        return self.get_state()
    
    # Calculaes all variables for current state
    def get_state(self):
        player_hand = self.player_hands[self.current_hand]
        player_total = self.hand_value(player_hand)
        dealer_upcard = card_value(self.dealer_hand[0])
        soft_hand = 'A' in player_hand and player_total <= 21
        can_split = can_split_hand(player_hand)
        decks_remaining = max(1, len(self.deck) / 52)
        true_count = int(self.running_count / decks_remaining)
        return (player_total, dealer_upcard, soft_hand, true_count, can_split)
    
    # Determines step of agent when passed the action of choice
    def step(self, action):
        player_hand = self.player_hands[self.current_hand]
        if action == "hit":
            player_hand.append(self.deal_card())
            if self.hand_value(player_hand) > 21:
                return self.get_state(), -1, True
        elif action == "double":
            player_hand.append(self.deal_card())
            if self.hand_value(player_hand) > 21:
                return self.get_state(), -1.5, True
            return self.resolve_dealer()
        elif action == "split" and can_split_hand(player_hand):
            self.player_hands.append([self.player_hands[self.current_hand].pop()])
            self.player_hands[self.current_hand].append(self.deal_card())
            self.player_hands[-1].append(self.deal_card())
            return self.get_state(), 0, False
        elif action == "stand":
            return self.resolve_dealer()
        return self.get_state(), 0, False
    
    # Handles dealers turn (follows rule that must hit under 17)
    def resolve_dealer(self):
        while self.hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self.deal_card())
        player_total = self.hand_value(self.player_hands[self.current_hand])
        dealer_total = self.hand_value(self.dealer_hand)
        if dealer_total > 21 or player_total > dealer_total:
            return self.get_state(), 1, True
        elif player_total == dealer_total:
            return self.get_state(), 0, True
        else:
            return self.get_state(), -1, True

class QLearning:
    def __init__(self):
        self.q_table = {}
    
    def get_q_vals(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
        return self.q_table[state]
    
    def choose_action(self, state, episode):
        epsilon = max(0.01, EPSILON * (0.90 ** (episode // 5000)))
        if random.uniform(0,1) < epsilon:
            return random.choice(ACTIONS)
        return ACTIONS[np.argmax(self.get_q_vals(state))]
    
    def update_q_table(self, state, action, reward, next_state):
        action_index = ACTIONS.index(action)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(ACTIONS))
        next_max_val = np.max(self.q_table[next_state])
        self.q_table[state][action_index] += ALPHA * (reward + GAMMA * next_max_val - self.q_table[state][action_index])
    
    def save_q_table(self, filename=Q_TABLE_FILE):
        with open(filename, 'wb') as file:
            pickle.dump(self.q_table, file)

    def load_q_table(self, filename=Q_TABLE_FILE):
        try:
            with open(filename, 'rb') as file:
                self.q_table = pickle.load(file)
        except FileNotFoundError:
            pass

env = BlackjackEnv()
agent = QLearning()
agent.load_q_table()

win_rates = []
push_rates = []
loss_rates = []

for episode in range(EPISODES):
    ALPHA = max(0.01, 0.1 * (0.99 ** (episode // 10000)))
    state = env.reset_table()
    done = False
    while not done:
        action = agent.choose_action(state, episode)
        next_state, reward, done = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        
    if episode % EVAL_INTERVAL == 0 and episode > 0:
        wins = 0
        pushes = 0
        losses = 0
        games = 10000
        for _ in range(games):
            state = env.reset_table()
            done = False
            while not done:
                action = agent.choose_action(state, EPISODES)
                state, reward, done = env.step(action)
            if reward > 0:
                wins+=1
            elif reward == 0:
                pushes+=1
            else:
                losses+=1
        
        win_rate = wins / games
        push_rate = pushes / games
        loss_rate = losses / games
        print(f'\n-------------------\nEpisode {episode}/{EPISODES}\n - Win Rate: {win_rate:.2%}\n - Push Rate: {push_rate:.2%}\n - Loss Rate: {loss_rate:.2%}\n')
        win_rates.append(win_rate)
        push_rates.append(push_rate)
        loss_rates.append(loss_rate)

agent.save_q_table()


episodes = np.arange(0, EPISODES, EVAL_INTERVAL)

episodes = episodes[:len(win_rates)]  # Ensure matching lengths

# Create the plot
plt.figure(figsize=(10, 6))

# Stacked area plot using `stackplot`
plt.stackplot(
    episodes, win_rates, push_rates, loss_rates,
    labels=["Win Rate", "Push Rate", "Loss Rate"],
    colors=["green", "blue", "red"],
    alpha=0.6
)

plt.xlabel('Episodes')
plt.ylabel('Rate')
plt.title('Win, Push, and Loss Rates Over Time')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()